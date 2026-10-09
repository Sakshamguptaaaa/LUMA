//! Teaching mode runtime: show one step, watch the user do it, check, repeat.
//!
//! OBSERVE → ASK (one move) → SHOW (point + say) → WATCH (thumbnail diffs
//! only) → when the screen changed and settled → OBSERVE …
//!
//! Pressing the shortcut pauses the lesson (epoch cancellation) so the user
//! can ask "why?"; it resumes by itself after that answer, or on "continue".
//! "Never mind" ends it. With `remember_lessons` on, progress (text only) is
//! saved so "continue from where we left off" works after a restart.

use crate::companion::{Companion, Phase};
use crate::screen::{self, CaptureOptions};
use crate::AppState;
use anyhow::{anyhow, Result};
use luma_core::annotation::{Annotation, ShapeKind};
use luma_core::lesson::{lesson_context, parse_move, split_reply, Lesson, LessonContext, Move, Step, LESSON_PROMPT};
use luma_net::gemini::ImagePart;
use luma_net::vision::{changed_fraction, thumbnail};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

/// The screen counts as "changed by the user" above this thumbnail diff.
const CHANGED: f64 = 0.008;
/// …and as settled once consecutive frames differ by less than this.
const STILL: f64 = 0.004;
/// Gentle nudge after this long with no change; pause after `GIVE_UP`.
const NUDGE: Duration = Duration::from_secs(60);
const GIVE_UP: Duration = Duration::from_secs(600);
const MAX_STEPS: u32 = 40;

fn lesson_path(app: &AppHandle) -> std::path::PathBuf {
    app.state::<AppState>().prefs_path.with_file_name("lesson.json")
}

/// Save (or forget) progress, only when the user opted in.
pub fn persist(app: &AppHandle, lesson: Option<&Lesson>) {
    let st = app.state::<AppState>();
    let path = lesson_path(app);
    match lesson {
        Some(l) if st.prefs.lock().unwrap().remember_lessons => {
            if let Ok(json) = serde_json::to_string_pretty(l) {
                let _ = std::fs::write(&path, json);
            }
        }
        _ => {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// A lesson saved by an earlier session.
pub fn load_saved(app: &AppHandle) -> Option<Lesson> {
    if !app.state::<AppState>().prefs.lock().unwrap().remember_lessons {
        return None;
    }
    std::fs::read_to_string(lesson_path(app)).ok().and_then(|s| serde_json::from_str(&s).ok())
}

/// Start or resume the lesson held by the companion.
pub async fn run(c: Arc<Companion>, app: AppHandle, epoch: u64, display: Option<usize>) {
    if c.lesson_running_swap(true) {
        return; // already running
    }
    let r = run_inner(&c, &app, epoch, display).await;
    c.lesson_running_swap(false);
    if let Err(e) = r {
        if c.current(epoch) {
            log::warn!("lesson failed: {e:#}");
            c.status(&app, Phase::Error, Some(format!("{e}")));
            c.speak(&app, epoch, "Something went wrong, so I paused the lesson. Say continue to try again.").await;
        }
    }
}

async fn run_inner(c: &Arc<Companion>, app: &AppHandle, epoch: u64, mut display: Option<usize>) -> Result<()> {
    let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
    let mut changed: Option<bool> = None;
    let mut invalid = 0;
    let mut waits = 0;
    loop {
        if !c.current(epoch) {
            return Ok(()); // paused by the shortcut; the lesson stays in memory
        }
        let Some(lesson) = c.lesson() else { return Ok(()) };
        if lesson.done.len() as u32 >= MAX_STEPS {
            c.speak(app, epoch, "We've done a lot of steps, so let's stop here. Say continue if you want to keep going.").await;
            c.finish_when_quiet(app, epoch);
            return Ok(());
        }
        c.status(app, Phase::Thinking, Some(lesson.status_line()));

        // ---- observe
        let (raw, snap) = {
            let app2 = app.clone();
            tauri::async_runtime::spawn_blocking(move || -> Result<_> {
                let displays = screen::displays()?;
                let pointer = screen::pointer(&app2, &displays);
                let raw = screen::capture(displays, pointer, display)?;
                let snap = raw.encode(&CaptureOptions { max_edge: 1600, closeup: false })?;
                Ok((raw, snap))
            })
            .await
            .map_err(|e| anyhow!("{e}"))??
        };
        display.get_or_insert(raw.display.index);
        if prefs.is_excluded(&snap.window.app) {
            c.speak(app, epoch, &format!("I paused the lesson because {} is in front, and I don't look at that app.", snap.window.app)).await;
            c.finish_when_quiet(app, epoch);
            return Ok(());
        }

        // ---- ask for one move
        let level = luma_core::prompt::level_instruction(app.state::<AppState>().session.lock().unwrap().level);
        let ctx = lesson_context(&LessonContext {
            lesson: &lesson,
            app: Some(snap.window.app.as_str()),
            window_title: Some(snap.window.title.as_str()),
            changed,
            level_instruction: level,
        });
        let model = c.llm_for(&prefs, "MEDIA_RESOLUTION_HIGH");
        let body = model.build_body(LESSON_PROMPT, &[], &ctx, &[ImagePart { jpeg: &snap.images[0].jpeg }], "What is the next move?");
        let reply = model.complete(&body).await?;
        if !c.current(epoch) {
            return Ok(());
        }
        let (speech, tag) = split_reply(&reply);
        let mv = tag
            .ok_or_else(|| "no lesson tag".to_string())
            .and_then(|t| parse_move(&t, lesson.step_number(), &snap.images[0].sent, &raw.display));
        let mv = match mv {
            Ok(m) => {
                invalid = 0;
                m
            }
            Err(e) => {
                invalid += 1;
                log::debug!("lesson reply unusable ({e}): {reply:?}");
                if invalid >= 3 {
                    return Err(anyhow!("I couldn't work out the next step"));
                }
                continue;
            }
        };
        log::info!("lesson move: {mv:?}");

        // ---- show
        match mv {
            Move::Done { summary } => {
                let _ = app.emit("luma://clear", ());
                let line = if speech.is_empty() { summary.clone() } else { speech };
                c.speak(app, epoch, &line).await;
                if let Some(mut l) = c.lesson() {
                    l.finish();
                    let _ = app.emit("luma://answer", format!("Lesson: {} — done in {} steps.", l.goal, l.done.len()));
                    record(app, &l, &summary);
                }
                c.set_lesson(None);
                persist(app, None);
                c.finish_when_quiet(app, epoch);
                return Ok(());
            }
            Move::Wait => {
                // Something is still loading or animating: look again shortly,
                // a few times, before waiting on the user again.
                waits += 1;
                if waits <= 4 {
                    tokio::time::sleep(Duration::from_millis(1500)).await;
                    continue;
                }
                waits = 0;
            }
            Move::Next(step) => {
                waits = 0;
                c.update_lesson(|l| l.advance(step.clone()));
                show_step(c, app, epoch, &step, &speech, prefs.annotate != "never").await;
            }
            Move::Retry { step, why } => {
                waits = 0;
                c.update_lesson(|l| l.retry(step.clone()));
                let line = if speech.is_empty() { why } else { speech };
                show_step(c, app, epoch, &step, &line, prefs.annotate != "never").await;
            }
        }
        if let Some(l) = c.lesson() {
            persist(app, Some(&l));
        }

        // ---- watch the user
        match watch(c, app, epoch, display).await {
            Watched::Changed => changed = Some(true),
            Watched::Stopped => return Ok(()),
            Watched::GaveUp => {
                c.speak(app, epoch, "I'll pause the lesson for now. Say continue whenever you want to pick it up.").await;
                c.finish_when_quiet(app, epoch);
                return Ok(());
            }
        }
    }
}

async fn show_step(c: &Arc<Companion>, app: &AppHandle, epoch: u64, step: &Step, line: &str, draw: bool) {
    let _ = app.emit("luma://clear", ());
    if draw {
        if let Some(at) = step.target {
            let to = format!("overlay-{}", at.display_index);
            let id = format!("_lesson{}", step.n);
            let label = step.label.clone().or_else(|| Some(step.instruction.clone()));
            let _ = app.emit_to(&to, "luma://annotate", Annotation::Shape { display: at.display_index, id: id.clone(), kind: ShapeKind::Box, rect: at.rect, label: None });
            let _ = app.emit_to(&to, "luma://annotate", Annotation::Step { display: at.display_index, id: format!("{id}n"), n: step.n, rect: at.rect, label: None });
            let _ = app.emit_to(&to, "luma://annotate", Annotation::Point { display: at.display_index, id: format!("{id}p"), rect: at.rect, label });
        }
    }
    let _ = app.emit("luma://step", serde_json::json!({ "step": step.n, "say": line, "action": step.instruction }));
    c.speak(app, epoch, line).await;
    c.status_when_quiet(app, epoch, Phase::Teaching, Some(format!("Your turn · Step {}: {}", step.n, step.instruction)));
}

enum Watched {
    Changed,
    Stopped,
    GaveUp,
}

/// Wait for the user to change the screen, then for it to settle.
async fn watch(c: &Arc<Companion>, app: &AppHandle, epoch: u64, display: Option<usize>) -> Watched {
    let started = Instant::now();
    let mut nudged = false;
    // Let LUMA's own speech start and the pointer land before the baseline.
    tokio::time::sleep(Duration::from_millis(600)).await;
    let Some(base) = thumb(display).await else { return Watched::Stopped };
    let key = screen::active_window().map(|w| w.key()).unwrap_or_default();
    loop {
        tokio::time::sleep(Duration::from_millis(350)).await;
        if !c.current(epoch) {
            return Watched::Stopped;
        }
        let Some(now) = thumb(display).await else { continue };
        let switched = screen::active_window().map(|w| w.key()).is_some_and(|k| !k.is_empty() && k != key);
        if switched || changed_fraction(&base, &now) > CHANGED {
            // settle: two consecutive quiet frames, at most ~4 s
            let mut prev = now;
            let t = Instant::now();
            while t.elapsed() < Duration::from_secs(4) {
                tokio::time::sleep(Duration::from_millis(300)).await;
                if !c.current(epoch) {
                    return Watched::Stopped;
                }
                let Some(n) = thumb(display).await else { break };
                let still = changed_fraction(&prev, &n) < STILL;
                prev = n;
                if still {
                    break;
                }
            }
            return Watched::Changed;
        }
        if !nudged && started.elapsed() > NUDGE && !c.is_speaking() {
            nudged = true;
            c.speak(app, epoch, "Take your time. If you're stuck, hold the shortcut and ask me.").await;
            if let Some(l) = c.lesson() {
                c.status_when_quiet(app, epoch, Phase::Teaching, Some(format!("Your turn · {}", l.status_line())));
            }
        }
        if started.elapsed() > GIVE_UP {
            return Watched::GaveUp;
        }
    }
}

async fn thumb(display: Option<usize>) -> Option<Vec<u8>> {
    tauri::async_runtime::spawn_blocking(move || screen::capture_display(display.unwrap_or(0)).ok().map(|i| thumbnail(&i)))
        .await
        .ok()
        .flatten()
}

fn record(app: &AppHandle, l: &Lesson, summary: &str) {
    let st = app.state::<AppState>();
    st.session.lock().unwrap().record(
        luma_core::session::Turn {
            user: format!("(lesson) {}", l.goal),
            assistant: format!("I taught the user: {}. Steps: {}. {summary}", l.goal, l.done.join("; ")),
            context_key: "lesson".into(),
        },
        std::iter::empty(),
    );
}
