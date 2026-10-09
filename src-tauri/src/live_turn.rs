//! Fast mode: a whole turn over Gemini Live (native audio).
//!
//! Shortcut down → capture + open the session while the user talks (the mic
//! audio is buffered meanwhile) → screenshot as the first user content →
//! `activityStart` + live audio → shortcut up → `activityEnd` → LUMA's voice
//! streams straight to the speaker; `draw` calls are resolved like tags and
//! drawn as they arrive.
//!
//! No separate speech-to-text or text-to-speech hop, so the answer starts
//! well under a second after the user stops talking. Tasks and lessons stay
//! on the standard pipeline; if Live is unreachable, LUMA falls back to it
//! for the next few minutes.

use crate::audio::{start_mic, MicHandle};
use crate::companion::{Companion, Phase};
use crate::screen::CaptureOptions;
use crate::settings::{get_key, Prefs, Provider};
use crate::AppState;
use anyhow::Result;
use luma_core::annotation::Resolver;
use luma_core::prompt::{context_block, live_system_prompt, TurnContext};
use luma_core::session::Turn;
use luma_core::speech::SpeechChunker;
use luma_net::live::{draw_call_to_tag, open, LiveConfig, LiveEvent};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::mpsc;

/// The microphone of a fast-mode turn; dropping it ends the user's speech.
pub struct LiveTurn {
    pub epoch: u64,
    pub mic: Option<MicHandle>,
    pub started: Instant,
}

/// After a Live failure, use the standard pipeline for this long.
pub const FALLBACK_FOR: Duration = Duration::from_secs(300);

/// Shortcut down in fast mode. Returns false if the mic is unavailable.
pub fn press(c: &Arc<Companion>, app: &AppHandle, prefs: &Prefs, epoch: u64) -> bool {
    let snapshot = c.spawn_capture(app, prefs, None);
    let (mic_tx, mic_rx) = mpsc::channel::<Vec<u8>>(512);
    let mic = match start_mic(mic_tx, c.level_emitter(app)) {
        Ok(m) => m,
        Err(e) => {
            snapshot.abort();
            c.status(app, Phase::Error, Some(e.to_string()));
            return false;
        }
    };
    *c.live_turn.lock().unwrap() = Some(LiveTurn { epoch, mic: Some(mic), started: Instant::now() });
    let c2 = c.clone();
    let app2 = app.clone();
    let prefs = prefs.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = run(&c2, &app2, epoch, snapshot, mic_rx, &prefs).await {
            if c2.current(epoch) {
                log::warn!("fast mode failed: {e:#}");
                *c2.live_failed_at.lock().unwrap() = Some(Instant::now());
                c2.status(&app2, Phase::Error, Some(format!("Fast mode isn't working right now ({e}). I'll use the standard voice for a few minutes; please ask again.")));
            }
        }
    });
    true
}

/// Shortcut up: closing the mic ends the user's turn.
pub fn release(c: &Arc<Companion>, app: &AppHandle) -> bool {
    let Some(mut t) = c.live_turn.lock().unwrap().take() else { return false };
    drop(t.mic.take());
    if t.started.elapsed() < Duration::from_millis(250) {
        c.status(app, Phase::Idle, Some("Hold the shortcut while you talk.".into()));
    } else {
        c.status(app, Phase::Thinking, None);
    }
    true
}

async fn run(
    c: &Arc<Companion>,
    app: &AppHandle,
    epoch: u64,
    snapshot: tauri::async_runtime::JoinHandle<Result<crate::screen::RawSnapshot>>,
    mut mic_rx: mpsc::Receiver<Vec<u8>>,
    prefs: &Prefs,
) -> Result<()> {
    let raw = match snapshot.await {
        Ok(Ok(r)) => Arc::new(r),
        Ok(Err(e)) => {
            if let Some(x) = e.downcast_ref::<crate::companion::Excluded>() {
                c.status(app, Phase::Idle, Some(x.to_string()));
                return Ok(());
            }
            return Err(e);
        }
        Err(_) => return Ok(()),
    };
    let raw2 = raw.clone();
    let snap = tauri::async_runtime::spawn_blocking(move || raw2.encode(&CaptureOptions { max_edge: 1600, closeup: false })).await??;
    let st = app.state::<AppState>();
    let draw = prefs.annotate != "never";
    let context_key = snap.context_key();
    let (system, prior_items, turn_no) = {
        let session = st.session.lock().unwrap();
        let marked = session.describe_items(&context_key, &snap.images[0].sent, &snap.displays);
        let focused = snap.focus.as_ref().and_then(|f| luma_core::prompt::describe_focus(&f.role, &f.name, f.secure));
        let ctx = context_block(&TurnContext {
            app: Some(snap.window.app.as_str()).filter(|s| !s.is_empty()),
            window_title: Some(snap.window.title.as_str()),
            pointer: snap.pointer_norm,
            pointer_closeup: None,
            has_closeup: false,
            display_count: snap.displays.len(),
            level: session.level,
            marked_items: &marked,
            focused: focused.as_deref(),
            selection: snap.focus.as_ref().and_then(|f| f.selection.as_deref()),
            lesson: None,
        });
        let mut history = String::new();
        for t in session.turns() {
            history += &format!("User: {}\nLUMA: {}\n", t.user, t.assistant);
        }
        let mut system = live_system_prompt(draw);
        if !history.is_empty() {
            system += &format!("\n## Conversation so far\n{history}");
        }
        system += &format!("\n{ctx}\nThe image in the conversation is the user's screen right now.");
        (system, session.items_for(&context_key), session.next_turn_number())
    };
    let cfg = LiveConfig {
        api_key: get_key(Provider::Gemini).unwrap_or_default(),
        model: prefs.live_model.clone(),
        system,
        draw_tool: draw,
    };
    let t_open = Instant::now();
    let (session, mut events) = open(&cfg).await?;
    log::info!("timing: live session ready in {} ms", t_open.elapsed().as_millis());
    if !c.current(epoch) {
        return Ok(());
    }
    session.send_image_turn(&snap.images[0].jpeg, None);
    session.activity_start();
    // Everything said so far was buffered in the channel; stream it, then
    // live audio until the shortcut comes up (the channel closes).
    let mut spoken_bytes = 0usize;
    while let Some(chunk) = mic_rx.recv().await {
        if !c.current(epoch) {
            return Ok(());
        }
        spoken_bytes += chunk.len();
        session.send_audio(&chunk);
    }
    session.activity_end();
    let t_end = Instant::now();
    if spoken_bytes < 2 * 16_000 / 4 {
        return Ok(()); // under a quarter second: release() already said why
    }

    let images: Vec<_> = snap.images.iter().map(|i| i.sent).collect();
    let mut resolver = Resolver::new(&snap.displays, &images, prior_items, turn_no);
    c.begin_marks(epoch);
    let mut improves = 0usize;
    let mut question = String::new();
    let mut said = String::new();
    let mut chunker = SpeechChunker::new();
    let mut samples = 0usize;
    let mut cleared = false;
    let mut drew = false;
    let mut nudges = 0;
    let deadline = tokio::time::sleep(Duration::from_secs(45));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            _ = &mut deadline => break,
            e = events.recv() => {
                if !c.current(epoch) {
                    return Ok(());
                }
                match e {
                    None => break,
                    Some(LiveEvent::Audio(pcm)) => {
                        if samples == 0 {
                            let ms = t_end.elapsed().as_millis();
                            log::info!("timing: live first audio {ms} ms after release");
                            let _ = app.emit("luma://timing", serde_json::json!({ "first_audio_ms": ms, "model": prefs.live_model }));
                            c.set_speaking(true);
                            c.status(app, Phase::Speaking, None);
                        }
                        samples += pcm.len();
                        c.play_pcm(pcm, luma_net::live::OUTPUT_RATE);
                        deadline.as_mut().reset(tokio::time::Instant::now() + Duration::from_secs(8));
                    }
                    Some(LiveEvent::InputText(t)) => {
                        question.push_str(&t);
                        let _ = app.emit("luma://transcript", question.trim());
                    }
                    Some(LiveEvent::OutputText(t)) => {
                        said.push_str(&t);
                        for u in chunker.push(&t) {
                            let _ = app.emit("luma://caption", &u);
                        }
                    }
                    Some(LiveEvent::ToolCall { id, name, args }) => {
                        session.tool_response(&id, &name);
                        if name != "draw" || !draw {
                            continue;
                        }
                        drew = true;
                        if !cleared {
                            cleared = true;
                            let _ = app.emit("luma://clear", ());
                        }
                        match draw_call_to_tag(&args).map(|t| resolver.resolve(&t)) {
                            Some(Ok(a)) => {
                                let a = if a.display() == Some(raw.display.index) {
                                    let a = luma_net::vision::place_board(a, &mut resolver, &raw.full, &raw.display);
                                    luma_net::vision::snap_sketch_to_ink(&a, &raw.full, &raw.display)
                                } else {
                                    a
                                };
                                c.show_annotation(app, &a);
                                if improves < crate::companion::MAX_IMPROVES {
                                    improves += 1;
                                    c.spawn_improve(app, epoch, raw.clone(), a.clone(), prefs);
                                }
                                for ready in resolver.take_ready() {
                                    c.show_annotation(app, &ready);
                                }
                            }
                            Some(Err(e)) => log::debug!("dropped draw <{}>: {}", e.tag, e.reason),
                            None => log::debug!("bad draw call: {args}"),
                        }
                    }
                    // Non-blocking draw calls split the answer into several
                    // model turns; it is over once nothing more arrives for a moment.
                    // Live sometimes ends a turn with nothing in it; ask again.
                    Some(LiveEvent::TurnComplete) if samples == 0 && !drew && nudges < 2 => {
                        nudges += 1;
                        log::debug!("live: empty turn, nudging");
                        session.nudge();
                    }
                    Some(LiveEvent::TurnComplete) => {
                        if samples > 0 {
                            deadline.as_mut().reset(tokio::time::Instant::now() + Duration::from_millis(1800));
                        }
                    }
                    Some(LiveEvent::Interrupted) => {}
                    Some(LiveEvent::Error(e)) => return Err(anyhow::anyhow!(e)),
                }
            }
        }
    }
    drop(session);
    if let Some(u) = chunker.finish() {
        let _ = app.emit("luma://caption", &u);
    }
    let question = question.trim().to_string();
    let answer = luma_core::speech::clean(&said).unwrap_or_default();
    if !question.is_empty() {
        let _ = app.emit("luma://question", &question);
    }
    if samples == 0 {
        c.status(app, Phase::Idle, Some("I didn't catch that.".into()));
        return Ok(());
    }
    let items: Vec<_> = resolver.items.values().filter(|m| m.turn == turn_no).cloned().collect();
    st.session.lock().unwrap().record(Turn { user: question, assistant: answer.clone(), context_key }, items);
    if !answer.is_empty() {
        let _ = app.emit("luma://answer", &answer);
    }
    c.finish_when_quiet(app, epoch);
    Ok(())
}

/// Whether this turn should use fast mode.
pub fn use_live(c: &Companion, prefs: &Prefs) -> bool {
    prefs.voice_engine == "live"
        && prefs.uses_gemini()
        && c.lesson().is_none()
        && !c.is_task_running()
        && c.live_failed_at.lock().unwrap().is_none_or(|t| t.elapsed() > FALLBACK_FOR)
}
