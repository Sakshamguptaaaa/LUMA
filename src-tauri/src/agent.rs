//! Doing tasks on the user's computer.
//!
//! OBSERVE → DECIDE (one action) → SHOW → CHECK POLICY → ACT → VERIFY → repeat
//!
//! * One action per model call, always on a fresh screenshot, so the model
//!   reacts to what actually happened instead of a plan made blind.
//! * Before acting, LUMA's pointer flies to the target and boxes it, so the
//!   user sees what is about to happen.
//! * Consequential actions pause for a spoken "yes"; secrets are never typed.
//! * After acting, the screen is compared with the "before" frame. The model
//!   is told explicitly when nothing changed, which is how failures get
//!   noticed and recovered from.
//! * Pressing the shortcut at any time stops the task (epoch cancellation).

use crate::companion::{Companion, Phase};
use crate::input;
use crate::screen::{self, CaptureOptions};
use crate::AppState;
use anyhow::{anyhow, Result};
use luma_core::action::{assess, parse_action, Action, ParsedAction, Risk, AGENT_TAGS};
use luma_core::annotation::{Annotation, ShapeKind};
use luma_core::markup::{MarkupParser, Segment};
use luma_core::prompt::{agent_context, AgentContext, AGENT_PROMPT};
use luma_core::session::Turn;
use luma_net::gemini::ImagePart;
use luma_net::vision::{changed_fraction, thumbnail};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

/// Everything needed to continue a task after a pause (approval / question).
#[derive(Debug, Clone)]
pub struct TaskState {
    pub goal: String,
    pub history: Vec<String>,
    pub display_index: Option<usize>,
    pub step: usize,
    pub last_changed: Option<bool>,
    /// App the task last acted in (for "undo that").
    pub app: String,
}

impl TaskState {
    pub fn new(goal: String, display_index: Option<usize>) -> Self {
        Self { goal, history: Vec::new(), display_index, step: 0, last_changed: None, app: String::new() }
    }
}

/// What the last task did, for "what did you change?" and "undo that".
#[derive(Debug, Clone, Default)]
pub struct TaskLog {
    pub goal: String,
    pub actions: Vec<String>,
    pub app: String,
    pub outcome: String,
}

#[derive(Debug, Clone)]
pub enum Pending {
    /// Waiting for "yes"/"no" before running `action`.
    Approval { task: TaskState, action: ParsedAction },
    /// Waiting for information the task needs.
    Answer { task: TaskState },
}

const CHANGE_THRESHOLD: f64 = 0.004;

fn platform() -> &'static str {
    if cfg!(target_os = "macos") {
        "macOS"
    } else if cfg!(target_os = "windows") {
        "Windows"
    } else {
        "Linux"
    }
}

/// Run (or resume) a task. `approved` is an action the user just OK'd.
pub async fn run(c: Arc<Companion>, app: AppHandle, epoch: u64, mut task: TaskState, approved: Option<ParsedAction>) {
    if !input::has_control_permission() {
        let e = input::request_control_permission();
        c.status(&app, Phase::Error, Some(e.to_string()));
        c.speak(&app, epoch, "I need Accessibility permission to do that. I've opened the settings for you.").await;
        return;
    }
    c.set_task_running(true);
    let result = run_inner(&c, &app, epoch, &mut task, approved).await;
    c.set_task_running(false);
    if !c.current(epoch) {
        return;
    }
    if let Err(e) = result {
        log::warn!("task failed: {e:#}");
        c.status(&app, Phase::Error, Some(format!("{e}")));
        c.speak(&app, epoch, "Something went wrong, so I stopped.").await;
        record(&app, &task, &format!("stopped with an error: {e}"));
    }
}

async fn run_inner(
    c: &Arc<Companion>,
    app: &AppHandle,
    epoch: u64,
    task: &mut TaskState,
    approved: Option<ParsedAction>,
) -> Result<()> {
    let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
    let max_steps = prefs.max_task_steps.max(3);
    c.status(app, Phase::Acting, Some(format!("Working on it: {}", task.goal)));

    if let Some(pa) = approved {
        let before = capture_thumb(app, task.display_index).await.ok();
        let note = execute(app, &pa.action, task.display_index).await?;
        task.history.push(with_note(pa.action.describe() + " (approved by the user)", note));
        task.last_changed = settle(app, task.display_index, before).await;
    }

    let mut invalid_in_a_row = 0;
    while task.step < max_steps {
        if !c.current(epoch) {
            return Ok(());
        }
        task.step += 1;
        let _ = app.emit("luma://clear", ());

        // ---- observe
        let (raw, snap) = {
            let app2 = app.clone();
            let prefer = task.display_index;
            tauri::async_runtime::spawn_blocking(move || -> Result<_> {
                let displays = screen::displays()?;
                let pointer = screen::pointer(&app2, &displays);
                let raw = screen::capture(displays, pointer, prefer)?;
                let snap = raw.encode(&CaptureOptions { max_edge: 1440, closeup: false })?;
                Ok((raw, snap))
            })
            .await
            .map_err(|e| anyhow!("{e}"))??
        };
        task.display_index.get_or_insert(raw.display.index);
        if prefs.is_excluded(&snap.window.app) {
            c.speak(app, epoch, &format!("I stopped because {} is in front, and I don't look at that app.", snap.window.app)).await;
            c.status(app, Phase::Idle, None);
            record(app, task, "stopped: excluded app in front");
            return Ok(());
        }
        let before = thumbnail(&raw.full);
        task.app = snap.window.app.clone();

        // ---- decide
        let ctx = agent_context(&AgentContext {
            goal: &task.goal,
            app: Some(snap.window.app.as_str()),
            window_title: Some(snap.window.title.as_str()),
            step: task.step,
            max_steps,
            history: &task.history,
            last_changed: task.last_changed,
            platform: platform(),
        });
        let model = c.llm_for(&prefs, "MEDIA_RESOLUTION_HIGH");
        let body = model.build_body(AGENT_PROMPT, &[], &ctx, &[ImagePart { jpeg: &snap.images[0].jpeg }], "Decide the next action.");
        let reply = model.complete(&body).await?;
        if !c.current(epoch) {
            return Ok(());
        }
        let (speech, tag) = split_reply(&reply);
        let Some(tag) = tag else {
            invalid_in_a_row += 1;
            task.history.push("(no action given; reply must end with exactly one action tag)".into());
            if invalid_in_a_row >= 2 {
                return Err(anyhow!("the model stopped giving actions"));
            }
            continue;
        };
        let pa = match parse_action(&tag, &snap.images[0].sent, &raw.display) {
            Ok(pa) => {
                invalid_in_a_row = 0;
                pa
            }
            Err(e) => {
                invalid_in_a_row += 1;
                task.history.push(format!("(invalid action: {e})"));
                if invalid_in_a_row >= 3 {
                    return Err(anyhow!("couldn't produce a valid action: {e}"));
                }
                continue;
            }
        };
        let _ = app.emit("luma://step", serde_json::json!({ "step": task.step, "say": speech, "action": pa.action.describe() }));

        // ---- terminal / conversational actions
        match &pa.action {
            Action::Done { summary } => {
                c.speak(app, epoch, summary).await;
                record(app, task, &format!("done: {summary}"));
                c.finish_when_quiet(app, epoch);
                return Ok(());
            }
            Action::Fail { reason } => {
                c.speak(app, epoch, reason).await;
                record(app, task, &format!("couldn't finish: {reason}"));
                c.finish_when_quiet(app, epoch);
                return Ok(());
            }
            Action::Ask { question } => {
                c.speak(app, epoch, question).await;
                c.set_pending(Pending::Answer { task: task.clone() });
                c.status(app, Phase::Waiting, Some(format!("{question} (hold the shortcut to answer)")));
                c.listen_for_reply_when_quiet(app, epoch);
                return Ok(());
            }
            _ => {}
        }

        // ---- show what is about to happen
        if !speech.is_empty() {
            c.say(app, epoch, &speech);
        }
        show_target(app, &pa.action, task.step);

        // ---- policy
        let context = format!("{} {}", snap.window.title, speech);
        match assess(&pa.action, pa.model_risk_high, &context) {
            Risk::Forbidden => {
                let q = "That needs a password or secret, which I never type. Please enter it yourself, then hold the shortcut and tell me to continue.";
                c.speak(app, epoch, q).await;
                task.history.push(format!("refused to {} (secret); asked the user to do it", pa.action.describe()));
                c.set_pending(Pending::Answer { task: task.clone() });
                c.status(app, Phase::Waiting, Some(q.into()));
                c.listen_for_reply_when_quiet(app, epoch);
                return Ok(());
            }
            Risk::NeedsApproval => {
                let q = format!("Should I go ahead and {}? Say yes or no.", approval_phrase(&pa.action));
                c.speak(app, epoch, &q).await;
                c.set_pending(Pending::Approval { task: task.clone(), action: pa });
                c.status(app, Phase::Waiting, Some(format!("{q} (hold the shortcut to answer)")));
                c.listen_for_reply_when_quiet(app, epoch);
                return Ok(());
            }
            Risk::Safe => {}
        }

        // ---- act + verify
        tokio::time::sleep(Duration::from_millis(450)).await; // let the user see the target
        if !c.current(epoch) {
            return Ok(());
        }
        let note = execute(app, &pa.action, task.display_index).await?;
        task.history.push(with_note(pa.action.describe(), note));
        task.last_changed = settle(app, task.display_index, Some(before)).await;
    }
    let msg = format!("I tried {max_steps} steps without finishing, so I stopped. Tell me if you want me to keep going.");
    c.speak(app, epoch, &msg).await;
    record(app, task, "stopped at the step limit");
    c.finish_when_quiet(app, epoch);
    Ok(())
}

fn with_note(line: String, note: Option<String>) -> String {
    match note {
        Some(n) => format!("{line} ({n})"),
        None => line,
    }
}

/// Speech is everything outside the first action tag.
fn split_reply(reply: &str) -> (String, Option<luma_core::markup::Tag>) {
    let mut p = MarkupParser::with_tags(AGENT_TAGS);
    let mut speech = String::new();
    let mut tag = None;
    for s in p.push(reply).into_iter().chain(p.finish()) {
        match s {
            Segment::Text(t) if tag.is_none() => speech.push_str(&t),
            Segment::Tag(t) if tag.is_none() => tag = Some(t),
            _ => {}
        }
    }
    (luma_core::speech::clean(&speech).unwrap_or_default(), tag)
}

fn approval_phrase(a: &Action) -> String {
    match a {
        Action::Click { label, .. } => format!("click \"{label}\""),
        Action::Type { text, .. } => format!("type \"{text}\""),
        Action::Key { keys } => format!("press {}", keys.join(" plus ")),
        other => other.describe(),
    }
}

fn show_target(app: &AppHandle, a: &Action, step: usize) {
    let (at, label) = match a {
        Action::Click { at, label, .. } => (Some(*at), Some(label.clone())),
        Action::Scroll { at: Some(at), .. } => (Some(*at), Some("Scrolling here".into())),
        _ => (None, None),
    };
    if let Some(at) = at {
        let id = format!("_step{step}");
        let _ = app.emit_to(
            format!("overlay-{}", at.display_index),
            "luma://annotate",
            Annotation::Shape { display: at.display_index, id: id.clone(), kind: ShapeKind::Box, rect: at.rect, label: None },
        );
        let _ = app.emit_to(
            format!("overlay-{}", at.display_index),
            "luma://annotate",
            Annotation::Point { display: at.display_index, id: format!("{id}p"), rect: at.rect, label },
        );
    }
}

/// Run one action. Returns a note for the history (how it was done, or what
/// verification found).
async fn execute(app: &AppHandle, a: &Action, display_index: Option<usize>) -> Result<Option<String>> {
    let a = a.clone();
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Option<String>> {
        let displays = screen::displays()?;
        let to_input = |r: &luma_core::geometry::DisplayRect| {
            displays
                .iter()
                .find(|d| d.index == r.display_index)
                .map(|d| d.view_to_input(r.rect.center()))
                .ok_or_else(|| anyhow!("display disappeared"))
        };
        let _ = (&app, display_index);
        match &a {
            Action::Click { at, button, double, .. } => {
                let p = to_input(at)?;
                // Accessibility press first: exact, and immune to overlapping
                // windows or a target that moved by a few pixels. The cursor
                // still glides there so the user sees what is happening.
                if *button == luma_core::action::MouseButton::Left && !*double {
                    input::glide_to(p)?;
                    if crate::ax::press_at(p) {
                        return Ok(Some("via accessibility".into()));
                    }
                }
                input::click(p, *button, *double)
            }
            Action::Type { text, clear_first } => {
                input::type_text(text, *clear_first)?;
                std::thread::sleep(Duration::from_millis(120));
                // Check the text landed in the focused field.
                return Ok(crate::ax::focused_value().map(|v| {
                    if v.contains(text.as_str()) {
                        "verified: the field now contains it".to_string()
                    } else {
                        "the focused field does NOT show this text; it may have gone elsewhere".to_string()
                    }
                }));
            }
            Action::Key { keys } => input::press_keys(keys),
            Action::Scroll { at, down, amount } => input::scroll(at.as_ref().map(&to_input).transpose()?, *down, *amount),
            Action::Open { url } => input::open_url(url),
            Action::Wait { ms } => {
                std::thread::sleep(Duration::from_millis(*ms));
                Ok(())
            }
            Action::Ask { .. } | Action::Done { .. } | Action::Fail { .. } => Ok(()),
        }
        .map(|()| None)
    })
    .await
    .map_err(|e| anyhow!("{e}"))?
}

async fn capture_thumb(app: &AppHandle, display_index: Option<usize>) -> Result<Vec<u8>> {
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Vec<u8>> {
        let displays = screen::displays()?;
        let idx = display_index
            .or_else(|| screen::pointer(&app, &displays).and_then(|p| luma_core::geometry::display_at(&displays, p)).map(|d| d.index))
            .unwrap_or(0);
        Ok(thumbnail(&screen::capture_display(idx)?))
    })
    .await
    .map_err(|e| anyhow!("{e}"))?
}

/// Wait for the screen to stop changing (≤ 3 s) and report whether it
/// changed compared with `before`.
async fn settle(app: &AppHandle, display_index: Option<usize>, before: Option<Vec<u8>>) -> Option<bool> {
    tokio::time::sleep(Duration::from_millis(500)).await;
    let start = Instant::now();
    let mut prev = capture_thumb(app, display_index).await.ok()?;
    while start.elapsed() < Duration::from_millis(2500) {
        tokio::time::sleep(Duration::from_millis(350)).await;
        let Ok(now) = capture_thumb(app, display_index).await else { break };
        let still = changed_fraction(&prev, &now) < CHANGE_THRESHOLD;
        prev = now;
        if still {
            break;
        }
    }
    before.map(|b| changed_fraction(&b, &prev) >= CHANGE_THRESHOLD)
}

fn record(app: &AppHandle, task: &TaskState, outcome: &str) {
    let steps = if task.history.is_empty() { "no actions".to_string() } else { task.history.join("; ") };
    let st = app.state::<AppState>();
    st.companion.set_last_task(TaskLog {
        goal: task.goal.clone(),
        actions: task.history.clone(),
        app: task.app.clone(),
        outcome: outcome.to_string(),
    });
    st.session.lock().unwrap().record(
        Turn {
            user: format!("(task) {}", task.goal),
            assistant: format!("I worked on: {}. Actions: {steps}. Result: {outcome}.", task.goal),
            context_key: "task".into(),
        },
        std::iter::empty(),
    );
    let _ = app.emit("luma://answer", format!("Task: {} — {outcome}. ({} steps)", task.goal, task.history.len()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_reply_takes_speech_and_first_action() {
        let (s, t) = split_reply(r#"Opening your GitHub settings. <open url="https://github.com/settings/admin"/> <click box="1 2 3 4"/>"#);
        assert_eq!(s, "Opening your GitHub settings.");
        let t = t.unwrap();
        assert_eq!(t.name, "open");
    }

    #[test]
    fn approval_phrases_read_naturally() {
        let a = Action::Key { keys: vec!["cmd".into(), "s".into()] };
        assert_eq!(approval_phrase(&a), "press cmd plus s");
    }
}
