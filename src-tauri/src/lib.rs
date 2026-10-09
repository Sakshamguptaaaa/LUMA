mod agent;
mod audio;
mod ax;
mod companion;
mod input;
mod lesson;
mod live_turn;
mod local_stt;
mod ocr;
mod screen;
mod settings;

use companion::{Companion, Phase};
use luma_core::geometry::Display;
use luma_core::session::Session;
use serde::Serialize;
use settings::{get_key, key_source, set_key, KeySource, Prefs, Provider};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

pub struct AppState {
    /// An update found by the updater, waiting for the user to install it.
    pub update: Mutex<Option<tauri_plugin_updater::Update>>,
    pub prefs: Mutex<Prefs>,
    pub prefs_path: PathBuf,
    pub session: Mutex<Session>,
    pub companion: Arc<Companion>,
}

#[derive(Serialize)]
struct SettingsView {
    prefs: Prefs,
    /// Where each configured key comes from. The keys themselves never reach the UI.
    keys: std::collections::BTreeMap<String, Option<KeySource>>,
    platform: &'static str,
}

#[tauri::command]
fn get_settings(state: tauri::State<AppState>) -> SettingsView {
    SettingsView {
        prefs: state.prefs.lock().unwrap().clone(),
        keys: Provider::ALL
            .iter()
            .map(|p| (format!("{p:?}").to_lowercase(), key_source(*p)))
            .collect(),
        platform: std::env::consts::OS,
    }
}

#[tauri::command]
fn save_key(state: tauri::State<AppState>, provider: Provider, key: String) -> Result<(), String> {
    set_key(provider, &key).map_err(|e| e.to_string())?;
    settings::apply_proxy(&state.prefs.lock().unwrap());
    Ok(())
}

#[tauri::command]
fn save_prefs(app: AppHandle, state: tauri::State<AppState>, prefs: Prefs) -> Result<(), String> {
    let old_hotkey = state.prefs.lock().unwrap().hotkey.clone();
    if prefs.hotkey != old_hotkey {
        register_hotkey(&app, &prefs.hotkey).map_err(|e| format!("couldn't use that shortcut: {e}"))?;
        let _ = app.global_shortcut().unregister(old_hotkey.as_str());
    }
    prefs.save(&state.prefs_path).map_err(|e| e.to_string())?;
    settings::apply_proxy(&prefs);
    *state.prefs.lock().unwrap() = prefs;
    refresh_tray(&app);
    Ok(())
}

#[tauri::command]
fn ask(app: AppHandle, state: tauri::State<AppState>, question: String) {
    if !question.trim().is_empty() {
        state.companion.ask_text(&app, question.trim().to_string());
    }
}

#[tauri::command]
fn stop(app: AppHandle, state: tauri::State<AppState>) {
    state.companion.cancel_all(&app);
    state.companion.status(&app, Phase::Idle, None);
}

#[tauri::command]
fn clear_session(app: AppHandle, state: tauri::State<AppState>) {
    state.companion.cancel_all(&app);
    state.session.lock().unwrap().clear();
    state.companion.status(&app, Phase::Idle, Some("Session cleared.".into()));
}

#[tauri::command]
fn set_paused(app: AppHandle, state: tauri::State<AppState>, paused: bool) {
    set_paused_inner(&app, &state, paused);
}

fn set_paused_inner(app: &AppHandle, state: &AppState, paused: bool) {
    if paused {
        state.companion.cancel_all(app);
    }
    {
        let mut p = state.prefs.lock().unwrap();
        p.paused = paused;
        let _ = p.save(&state.prefs_path);
    }
    state.companion.status(app, if paused { Phase::Paused } else { Phase::Idle }, None);
    refresh_tray(app);
}

pub fn show_panel(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

fn register_hotkey(app: &AppHandle, accel: &str) -> Result<(), String> {
    app.global_shortcut().register(accel).map_err(|e| e.to_string())
}

#[derive(Clone, Serialize, PartialEq)]
struct CursorEvent {
    display: usize,
    x: f64,
    y: f64,
}

#[derive(Clone, Serialize)]
struct HoldEvent {
    display: usize,
    x: f64,
    y: f64,
    /// 0..1 while charging, 1 when fired, -1 when cancelled.
    progress: f64,
}

/// Mouse tracking for the companion cursor, plus the point-and-ask gesture.
///
/// Reads the pointer and button state straight from the OS (~60 Hz, any
/// thread, no permission needed) and emits only on change. A press held
/// still for `long_press_ms` on someone else's app triggers a hands-free,
/// drawing turn about that spot; a filling ring shows it charging.
fn spawn_cursor_tracker(app: AppHandle) {
    std::thread::Builder::new()
        .name("luma-cursor".into())
        .spawn(move || {
            let mut displays = screen::displays().unwrap_or_default();
            let mut refreshed = std::time::Instant::now();
            let mut last: Option<CursorEvent> = None;
            let mut hidden = false;
            // (started, where, fired, shown)
            let mut press: Option<(std::time::Instant, luma_core::geometry::Point, bool, bool)> = None;
            let mut last_hold_emit = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(16));
                let st = app.state::<AppState>();
                let (paused, gesture, hold_ms) = {
                    let p = st.prefs.lock().unwrap();
                    (p.paused, p.gesture, p.long_press_ms.max(600))
                };
                if paused {
                    if !hidden {
                        let _ = app.emit("luma://cursor", Option::<CursorEvent>::None);
                        hidden = true;
                        last = None;
                    }
                    continue;
                }
                hidden = false;
                if refreshed.elapsed().as_secs() >= 3 || displays.is_empty() {
                    if let Ok(d) = screen::displays() {
                        displays = d;
                    }
                    refreshed = std::time::Instant::now();
                }
                let Some(p) = screen::pointer(&app, &displays) else { continue };
                let Some(d) = luma_core::geometry::display_at(&displays, p) else { continue };
                let v = d.input_to_view(p);
                let ev = CursorEvent { display: d.index, x: v.x.round(), y: v.y.round() };
                if last.as_ref() != Some(&ev) {
                    let _ = app.emit("luma://cursor", Some(&ev));
                    last = Some(ev);
                }

                // ---- long-press gesture
                if !gesture {
                    continue;
                }
                let down = screen::left_button_down();
                match (&mut press, down) {
                    (None, true) => press = Some((std::time::Instant::now(), p, false, false)),
                    (Some((t0, at, fired, shown)), true) => {
                        let moved = ((p.x - at.x).powi(2) + (p.y - at.y).powi(2)).sqrt() / d.input_per_point;
                        if moved > 8.0 {
                            if *shown {
                                let _ = app.emit("luma://hold", HoldEvent { display: d.index, x: v.x, y: v.y, progress: -1.0 });
                            }
                            press = None; // a drag, not a long-press
                            continue;
                        }
                        let ms = t0.elapsed().as_millis() as u64;
                        if *fired || ms < 350 {
                            continue;
                        }
                        let busy = st.companion.is_busy() || own_window_focused(&app);
                        if busy {
                            continue;
                        }
                        let av = d.input_to_view(*at);
                        if ms >= hold_ms {
                            *fired = true;
                            let _ = app.emit("luma://hold", HoldEvent { display: d.index, x: av.x, y: av.y, progress: 1.0 });
                            st.companion.on_gesture(&app, *at);
                        } else if last_hold_emit.elapsed().as_millis() >= 40 {
                            *shown = true;
                            last_hold_emit = std::time::Instant::now();
                            let progress = (ms - 350) as f64 / (hold_ms - 350) as f64;
                            let _ = app.emit("luma://hold", HoldEvent { display: d.index, x: av.x, y: av.y, progress });
                        }
                    }
                    (Some((_, _, fired, shown)), false) => {
                        if *shown && !*fired {
                            let _ = app.emit("luma://hold", HoldEvent { display: d.index, x: v.x, y: v.y, progress: -1.0 });
                        }
                        press = None;
                    }
                    (None, false) => {}
                }
            }
        })
        .expect("cursor thread");
}

/// Long-presses inside LUMA's own panel are just clicks.
fn own_window_focused(app: &AppHandle) -> bool {
    app.get_webview_window("main").and_then(|w| w.is_focused().ok()).unwrap_or(false)
}

/// Draws a box exactly around the active window and marks the pointer. If the
/// box hugs the window edges, coordinate mapping is correct on this setup.
fn check_alignment(app: &AppHandle) {
    use luma_core::annotation::{Annotation, ShapeKind};
    use luma_core::geometry::{display_at, Rect};
    let app = app.clone();
    std::thread::spawn(move || {
        // give the menu time to close and focus to return to the user's window
        std::thread::sleep(std::time::Duration::from_millis(700));
        let Ok(displays) = screen::displays() else { return };
        sync_overlays(&app, &displays);
        let _ = app.emit("luma://clear", ());
        if let Some(w) = screen::active_window() {
            if let Some(f) = w.frame {
                if let Some(d) = display_at(&displays, f.center()) {
                    let tl = d.input_to_view(luma_core::geometry::Point::new(f.x, f.y));
                    let rect = Rect::new(tl.x, tl.y, f.w / d.input_per_point, f.h / d.input_per_point)
                        .clamp_to(&d.view_bounds());
                    let _ = app.emit_to(
                        format!("overlay-{}", d.index),
                        "luma://annotate",
                        Annotation::Shape {
                            display: d.index,
                            id: "_align_window".into(),
                            kind: ShapeKind::Box,
                            rect,
                            label: Some(format!("{} window: box should hug its edges", w.app)),
                        },
                    );
                }
            }
        }
        if let Some(p) = screen::pointer(&app, &displays) {
            if let Some(d) = display_at(&displays, p) {
                let v = d.input_to_view(p);
                let _ = app.emit_to(
                    format!("overlay-{}", d.index),
                    "luma://annotate",
                    Annotation::Shape {
                        display: d.index,
                        id: "_align_pointer".into(),
                        kind: ShapeKind::Circle,
                        rect: Rect::new(v.x - 12.0, v.y - 12.0, 24.0, 24.0),
                        label: Some("your mouse pointer".into()),
                    },
                );
            }
        }
        let summary: Vec<String> = displays
            .iter()
            .map(|d| {
                let (w, h) = d.view_size();
                format!("{}: {w:.0}×{h:.0} pt @{:.1}x at ({:.0},{:.0})", d.name, d.scale_factor, d.input_frame.x, d.input_frame.y)
            })
            .collect();
        let _ = app.emit("luma://notice", format!("Alignment check. Displays: {}", summary.join("; ")));
        std::thread::sleep(std::time::Duration::from_secs(8));
        let _ = app.emit("luma://clear", ());
    });
}

/// Clears annotations once they no longer describe what is on screen: the
/// user switched app/window/tab, or the content scrolled or changed. Uses a
/// 64×40 thumbnail diff, only while marks are visible and LUMA is not talking
/// (live captions would otherwise count as changes on systems where overlays
/// are captured).
fn spawn_annotation_watcher(app: AppHandle) {
    std::thread::Builder::new()
        .name("luma-watch".into())
        .spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_millis(400));
            let st = app.state::<AppState>();
            let c = &st.companion;
            if c.is_task_running() {
                continue;
            }
            let (display, key, baseline, quiet_for) = {
                let w = c.watch.lock().unwrap();
                let Some(w) = w.as_ref() else { continue };
                (w.display, w.window_key.clone(), w.baseline.clone(), w.last_mark.elapsed())
            };
            let now_key = screen::active_window().map(|w| w.key()).unwrap_or_default();
            let mut stale = !key.is_empty() && !now_key.is_empty() && now_key != key;
            if !stale && quiet_for.as_millis() > 600 {
                if let Ok(img) = screen::capture_display(display) {
                    let thumb = luma_net::vision::thumbnail(&img);
                    match baseline {
                        None => {
                            if let Some(w) = c.watch.lock().unwrap().as_mut() {
                                w.baseline = Some(thumb);
                            }
                        }
                        Some(b) if !c.is_speaking() => {
                            stale = luma_net::vision::changed_fraction(&b, &thumb) > 0.06;
                        }
                        Some(_) => {}
                    }
                }
            }
            if stale {
                *c.watch.lock().unwrap() = None;
                let _ = app.emit("luma://clear", ());
            }
        })
        .expect("watch thread");
}

/// One transparent, click-through overlay window per display. Called at
/// startup and at every turn, so plugging in or rearranging monitors is
/// picked up without a restart.
pub fn sync_overlays(app: &AppHandle, displays: &[Display]) {
    let displays = displays.to_vec();
    let app2 = app.clone();
    let _ = app.run_on_main_thread(move || {
        for d in &displays {
            let label = format!("overlay-{}", d.index);
            let win = match app2.get_webview_window(&label) {
                Some(w) => w,
                None => {
                    let built = WebviewWindowBuilder::new(
                        &app2,
                        &label,
                        WebviewUrl::App(format!("overlay.html?display={}", d.index).into()),
                    )
                    .title("LUMA overlay")
                    .transparent(true)
                    .decorations(false)
                    .always_on_top(true)
                    .skip_taskbar(true)
                    .resizable(false)
                    .focused(false)
                    .shadow(false)
                    .visible_on_all_workspaces(true)
                    // Keep our own drawings out of the screenshots we send.
                    .content_protected(true)
                    .build();
                    match built {
                        Ok(w) => {
                            let _ = w.set_ignore_cursor_events(true);
                            elevate_overlay(&w);
                            w
                        }
                        Err(e) => {
                            log::error!("overlay window: {e}");
                            continue;
                        }
                    }
                }
            };
            let f = d.input_frame;
            if cfg!(target_os = "macos") {
                let _ = win.set_position(tauri::LogicalPosition::new(f.x, f.y));
                let _ = win.set_size(tauri::LogicalSize::new(f.w, f.h));
            } else {
                let _ = win.set_position(tauri::PhysicalPosition::new(f.x as i32, f.y as i32));
                let _ = win.set_size(tauri::PhysicalSize::new(f.w as u32, f.h as u32));
            }
            let _ = win.show();
        }
        // displays that went away
        for (label, w) in app2.webview_windows() {
            if let Some(i) = label.strip_prefix("overlay-").and_then(|s| s.parse::<usize>().ok()) {
                if i >= displays.len() {
                    let _ = w.close();
                }
            }
        }
    });
}

/// Float above the menu bar and full-screen apps, on every Space.
#[cfg(target_os = "macos")]
fn elevate_overlay(w: &tauri::WebviewWindow) {
    use objc2::runtime::AnyObject;
    if let Ok(ns) = w.ns_window() {
        let ns = ns as *mut AnyObject;
        // NSPopUpMenuWindowLevel (101): above the menu bar, below alerts.
        // Behaviour: canJoinAllSpaces | stationary | ignoresCycle | fullScreenAuxiliary
        let behaviour: usize = (1 << 0) | (1 << 4) | (1 << 6) | (1 << 8);
        unsafe {
            let _: () = objc2::msg_send![ns, setLevel: 101isize];
            let _: () = objc2::msg_send![ns, setCollectionBehavior: behaviour];
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn elevate_overlay(_w: &tauri::WebviewWindow) {}

/// Signed updates from GitHub Releases: checked shortly after launch and
/// every 6 hours. Never installed silently; the tray offers it.
fn spawn_update_checker(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(20)).await;
        loop {
            check_for_update(&app, false).await;
            tokio::time::sleep(std::time::Duration::from_secs(6 * 3600)).await;
        }
    });
}

async fn check_for_update(app: &AppHandle, tell: bool) {
    use tauri_plugin_updater::UpdaterExt;
    let found = match app.updater() {
        Ok(u) => u.check().await,
        Err(e) => Err(e),
    };
    match found {
        Ok(Some(u)) => {
            log::info!("update available: {}", u.version);
            let _ = app.emit("luma://notice", format!("LUMA {} is available. Install it from the menu bar icon.", u.version));
            *app.state::<AppState>().update.lock().unwrap() = Some(u);
            refresh_tray(app);
        }
        Ok(None) => {
            if tell {
                let _ = app.emit("luma://notice", "LUMA is up to date.");
            }
        }
        Err(e) => {
            log::debug!("update check failed: {e}");
            if tell {
                let _ = app.emit("luma://notice", "Couldn't check for updates right now.");
            }
        }
    }
}

async fn update_clicked(app: &AppHandle) {
    let pending = app.state::<AppState>().update.lock().unwrap().take();
    let Some(update) = pending else {
        show_panel(app);
        return check_for_update(app, true).await;
    };
    let _ = app.emit("luma://notice", format!("Downloading LUMA {}…", update.version));
    match update.download_and_install(|_, _| {}, || {}).await {
        Ok(()) => app.restart(),
        Err(e) => {
            log::warn!("update failed: {e}");
            let _ = app.emit("luma://notice", format!("The update didn't install: {e}"));
        }
    }
}

fn refresh_tray(app: &AppHandle) {
    let Some(tray) = app.tray_by_id("luma") else { return };
    if let Ok(menu) = build_tray_menu(app) {
        let _ = tray.set_menu(Some(menu));
    }
}

fn build_tray_menu(app: &AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let st = app.state::<AppState>();
    let prefs = st.prefs.lock().unwrap().clone();
    let hint = MenuItem::with_id(app, "hint", format!("Hold {} and talk", prefs.hotkey), false, None::<&str>)?;
    let open = MenuItem::with_id(app, "open", "Open LUMA…", true, None::<&str>)?;
    let pause = MenuItem::with_id(
        app,
        "pause",
        if prefs.paused { "Resume watching" } else { "Pause (stop seeing & listening)" },
        true,
        None::<&str>,
    )?;
    let clear = MenuItem::with_id(app, "clear", "Forget this session", true, None::<&str>)?;
    let update = st.update.lock().unwrap().as_ref().map(|u| format!("Install LUMA {} and restart", u.version));
    let update = MenuItem::with_id(app, "update", update.as_deref().unwrap_or("Check for updates"), true, None::<&str>)?;
    let align = MenuItem::with_id(app, "align", "Check overlay alignment", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit LUMA", true, None::<&str>)?;
    Menu::with_items(
        app,
        &[&hint, &PredefinedMenuItem::separator(app)?, &open, &pause, &clear, &align, &update, &PredefinedMenuItem::separator(app)?, &quit],
    )
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("luma=info,luma_lib=info")).init();

    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    let st = app.state::<AppState>();
                    match event.state() {
                        ShortcutState::Pressed => st.companion.on_press(app),
                        ShortcutState::Released => st.companion.on_release(app),
                    }
                })
                .build(),
        )
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let config_dir = app.path().app_config_dir()?;
            settings::load_env_files(Some(&config_dir));
            let prefs_path = config_dir.join("prefs.json");
            let prefs = Prefs::load(&prefs_path);
            settings::apply_proxy(&prefs);
            let hotkey = prefs.hotkey.clone();
            app.manage(AppState {
                update: Mutex::new(None),
                prefs: Mutex::new(prefs),
                prefs_path,
                session: Mutex::new(Session::new()),
                companion: Arc::new(Companion::new()),
            });
            let handle = app.handle().clone();
            if let Err(e) = register_hotkey(&handle, &hotkey) {
                log::error!("hotkey {hotkey} unavailable: {e}");
            }

            let menu = build_tray_menu(&handle)?;
            TrayIconBuilder::with_id("luma")
                .icon(app.default_window_icon().cloned().expect("app icon"))
                .icon_as_template(true)
                .tooltip("LUMA")
                .menu(&menu)
                .show_menu_on_left_click(true)
                .on_menu_event(|app, ev| {
                    let st = app.state::<AppState>();
                    match ev.id().as_ref() {
                        "open" => show_panel(app),
                        "pause" => {
                            let paused = !st.prefs.lock().unwrap().paused;
                            set_paused_inner(app, &st, paused);
                        }
                        "clear" => {
                            st.companion.cancel_all(app);
                            st.session.lock().unwrap().clear();
                        }
                        "align" => check_alignment(app),
                        "update" => {
                            let app = app.clone();
                            tauri::async_runtime::spawn(async move { update_clicked(&app).await });
                        }
                        "quit" => app.exit(0),
                        _ => {}
                    }
                })
                .on_tray_icon_event(|tray, ev| {
                    if let TrayIconEvent::Click { button: MouseButton::Right, button_state: MouseButtonState::Up, .. } = ev {
                        show_panel(tray.app_handle());
                    }
                })
                .build(app)?;

            match screen::displays() {
                Ok(d) => sync_overlays(&handle, &d),
                Err(e) => log::error!("{e}"),
            }
            spawn_update_checker(handle.clone());
            {
                use tauri::Listener;
                handle.listen("luma://overlay-health", |e| log::info!("overlay health: {}", e.payload()));
            }
            spawn_cursor_tracker(handle.clone());
            spawn_annotation_watcher(handle.clone());
            if !screen::has_screen_permission() {
                log::warn!("Screen Recording permission missing; LUMA will ask on first use");
            }
            let llm_key = handle.state::<AppState>().prefs.lock().unwrap().llm_key();
            if llm_key.into_iter().chain([Provider::Assemblyai]).any(|p| get_key(p).is_none()) {
                show_panel(&handle);
            }
            let _ = handle.emit("luma://status", companion::StatusEvent { phase: Phase::Idle, message: None, display: None });
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the panel hides it; LUMA keeps running in the menu bar.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![get_settings, save_key, save_prefs, ask, stop, clear_session, set_paused])
        .run(tauri::generate_context!())
        .expect("error while running LUMA");
}
