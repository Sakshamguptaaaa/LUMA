//! The turn loop: hotkey → listen → see → think → speak (+ draw) (+ act).
//!
//! Modes (decided when the shortcut is released):
//! * **Voice** — quick press: small screenshot, medium media resolution, no
//!   drawing. Cheapest per turn.
//! * **Annotate** — held ≥ `long_press_ms` (or "always"): full-resolution
//!   screenshot plus pointer close-up, boxes/arrows/labels on screen.
//! * **Lesson** — asked to do or learn something, the model answers with
//!   `<lesson goal=…/>` and LUMA guides the user one step at a time
//!   (`lesson.rs`); the user does the clicking.
//! * **Task** — only with autopilot on, the model may answer with
//!   `<task goal=…/>`, which hands off to the agent loop (`agent.rs`).
//!
//! Latency: capture and the STT connection start the moment the key goes
//! down; Gemini streams; speech is synthesized per utterance while the model
//! is still writing; annotations are synchronized through the audio queue.
//! Barge-in: every turn carries an epoch; pressing the key bumps it.

use crate::agent::{self, Pending, TaskState};
use crate::audio::{start_mic, MicHandle, Speaker, SpeakerCmd};
use crate::screen::{self, CaptureOptions, RawSnapshot, Snapshot};
use crate::settings::{get_key, Prefs, Provider};
use crate::AppState;
use anyhow::{anyhow, Result};
use luma_core::action::{yes_no, YesNo};
use luma_core::annotation::{Annotation, Resolver};
use luma_core::command::{local_command, LocalCommand};
use luma_core::markup::{MarkupParser, Segment};
use luma_core::prompt::{context_block, system_prompt, Draw, TurnContext};
use luma_core::session::Turn;
use luma_core::speech::SpeechChunker;
use luma_net::assemblyai::{self, SttEvent};
use luma_net::gemini::{Gemini, HistoryTurn, ImagePart};
use luma_net::sarvam::SarvamTts;
use luma_net::sarvam_ws::{self, TtsConfig, TtsEvent, TtsStream};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::async_runtime::JoinHandle;
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Idle,
    Listening,
    Thinking,
    Speaking,
    /// Carrying out a task on the computer.
    Acting,
    /// A task is paused, waiting for the user's yes/no or an answer.
    Waiting,
    /// Teaching: the user is doing a step; LUMA is watching.
    Teaching,
    Paused,
    Error,
}

#[derive(Clone, Serialize)]
pub struct StatusEvent {
    pub phase: Phase,
    pub message: Option<String>,
    /// Display whose overlay should show the HUD.
    pub display: Option<usize>,
}

/// Streaming voice connection for one turn.
pub struct VoiceConn {
    stream: TtsStream,
    events: mpsc::UnboundedReceiver<TtsEvent>,
}

#[derive(Clone)]
enum Out {
    Speech(String),
    Mark(Marker),
}

#[derive(Debug, Clone)]
enum Marker {
    Annotate(Annotation),
    Caption(String),
    /// Native-resolution inset for `<zoom>` (JPEG data URL).
    Magnify { display: usize, rect: luma_core::geometry::Rect, src: String },
}

/// Refined geometry for this turn's small marks, which may arrive before or
/// after the mark itself is shown.
#[derive(Default)]
struct RefineBook {
    epoch: u64,
    shown: std::collections::HashSet<String>,
    better: std::collections::HashMap<String, Annotation>,
}

/// Start the fast model alongside if the accurate one is still silent.
const HEDGE_AFTER_MS: u64 = 6000;

/// Say "one moment" if the model has been silent this long.
const SILENCE_FILLER_MS: u64 = 4000;

/// At most this many marks are snapped / refined per answer.
pub const MAX_IMPROVES: usize = 10;

struct Listening {
    epoch: u64,
    mic: Option<MicHandle>,
    stt: JoinHandle<Result<String>>,
    snapshot: JoinHandle<Result<RawSnapshot>>,
    voice: Option<JoinHandle<Option<VoiceConn>>>,
    started: Instant,
}

/// How a turn looks at the screen and how it may answer.
#[derive(Debug, Clone)]
pub struct TurnStyle {
    pub draw: Draw,
    pub max_edge: u32,
    pub closeup: bool,
    pub media: &'static str,
    pub model: String,
    pub thinking: String,
}

impl TurnStyle {
    /// Shortcut turn: fast model, smaller image; draws only when asked.
    pub fn quick(p: &Prefs) -> Self {
        Self {
            draw: match p.annotate.as_str() {
                "never" => Draw::Never,
                "always" => Draw::Always,
                _ => Draw::WhenUseful,
            },
            max_edge: 1280,
            closeup: p.send_closeup,
            media: "MEDIA_RESOLUTION_MEDIUM",
            model: p.fast_model.clone(),
            thinking: p.fast_thinking.clone(),
        }
    }

    /// Long-press / typed turn: accurate model, full detail, always draws.
    pub fn pointing(p: &Prefs) -> Self {
        Self {
            draw: if p.annotate == "never" { Draw::Never } else { Draw::Always },
            max_edge: p.max_image_edge,
            closeup: p.send_closeup,
            media: "MEDIA_RESOLUTION_HIGH",
            model: p.gemini_model.clone(),
            thinking: p.thinking_level.clone(),
        }
    }
}

/// Annotations currently on screen, and what the screen looked like then.
/// The watcher clears them when the user switches window or the content
/// changes (scroll, next slide, navigation).
pub struct Watch {
    pub display: usize,
    pub window_key: String,
    pub baseline: Option<Vec<u8>>,
    pub last_mark: Instant,
}

pub struct Companion {
    epoch: AtomicU64,
    speaker: Option<Speaker>,
    http: reqwest::Client,
    listening: Mutex<Option<Listening>>,
    hud_display: Mutex<Option<usize>>,
    pending: Mutex<Option<Pending>>,
    task_running: AtomicBool,
    speaking: AtomicBool,
    pub watch: Mutex<Option<Watch>>,
    /// The last answer's output (speech, captions, drawings), for "repeat".
    last_reply: Mutex<Vec<Out>>,
    refined: Mutex<RefineBook>,
    /// The lesson in progress or paused (teaching mode).
    lesson: Mutex<Option<luma_core::lesson::Lesson>>,
    lesson_running: AtomicBool,
    last_task: Mutex<Option<agent::TaskLog>>,
    /// A speech-to-text session opened after a turn, for an instant follow-up.
    warm_stt: Mutex<Option<assemblyai::SttConn>>,
    /// Fast mode (Gemini Live): the turn being spoken, and the last failure.
    pub live_turn: Mutex<Option<crate::live_turn::LiveTurn>>,
    pub live_failed_at: Mutex<Option<Instant>>,
}

/// A warm STT session is used only if it is younger than this.
const WARM_STT_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(6);

/// The foreground app was on the exclusion list.
#[derive(Debug)]
pub struct Excluded(pub String);
impl std::fmt::Display for Excluded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LUMA doesn't look at the screen while {} is in front.", self.0)
    }
}
impl std::error::Error for Excluded {}

impl Companion {
    pub fn new() -> Self {
        let speaker = match Speaker::spawn() {
            Ok(s) => Some(s),
            Err(e) => {
                log::error!("audio output unavailable: {e}");
                None
            }
        };
        Self {
            epoch: AtomicU64::new(0),
            speaker,
            http: reqwest::Client::builder()
                .pool_idle_timeout(std::time::Duration::from_secs(90))
                .build()
                .expect("http client"),
            listening: Mutex::new(None),
            hud_display: Mutex::new(None),
            pending: Mutex::new(None),
            task_running: AtomicBool::new(false),
            speaking: AtomicBool::new(false),
            watch: Mutex::new(None),
            last_reply: Mutex::new(Vec::new()),
            refined: Mutex::new(RefineBook::default()),
            lesson: Mutex::new(None),
            lesson_running: AtomicBool::new(false),
            last_task: Mutex::new(None),
            warm_stt: Mutex::new(None),
            live_turn: Mutex::new(None),
            live_failed_at: Mutex::new(None),
        }
    }

    pub fn current(&self, epoch: u64) -> bool {
        self.epoch.load(Ordering::SeqCst) == epoch
    }

    pub fn is_task_running(&self) -> bool {
        self.task_running.load(Ordering::SeqCst)
    }

    pub fn set_task_running(&self, v: bool) {
        self.task_running.store(v, Ordering::SeqCst);
    }

    /// Listening, carrying out a task, or waiting on the user: a long-press
    /// should not start a new turn.
    pub fn is_busy(&self) -> bool {
        self.listening.lock().unwrap().is_some() || self.is_task_running() || self.pending.lock().unwrap().is_some()
    }

    pub fn set_last_task(&self, t: agent::TaskLog) {
        *self.last_task.lock().unwrap() = Some(t);
    }

    pub fn lesson(&self) -> Option<luma_core::lesson::Lesson> {
        self.lesson.lock().unwrap().clone()
    }

    pub fn set_lesson(&self, l: Option<luma_core::lesson::Lesson>) {
        *self.lesson.lock().unwrap() = l;
    }

    pub fn update_lesson(&self, f: impl FnOnce(&mut luma_core::lesson::Lesson)) {
        if let Some(l) = self.lesson.lock().unwrap().as_mut() {
            f(l);
        }
    }

    /// Set the running flag; returns the previous value.
    pub fn lesson_running_swap(&self, v: bool) -> bool {
        self.lesson_running.swap(v, Ordering::SeqCst)
    }

    /// Resume a paused lesson (or one saved by an earlier session).
    /// Returns false when there is none.
    fn resume_lesson(self: &Arc<Self>, app: &AppHandle, epoch: u64) -> bool {
        if self.lesson().is_none() {
            match crate::lesson::load_saved(app) {
                Some(l) => self.set_lesson(Some(l)),
                None => return false,
            }
        }
        let display = *self.hud_display.lock().unwrap();
        tauri::async_runtime::spawn(crate::lesson::run(self.clone(), app.clone(), epoch, display));
        true
    }

    pub fn set_speaking(&self, v: bool) {
        self.speaking.store(v, Ordering::SeqCst);
    }

    /// Queue streamed PCM on the speaker (fast mode).
    pub fn play_pcm(&self, pcm: Vec<i16>, rate: u32) {
        if let Some(s) = &self.speaker {
            s.send(SpeakerCmd::Pcm(pcm, rate));
        }
    }

    /// Draw now (fast mode draws as calls arrive; the audio is already streaming).
    pub fn show_annotation(&self, app: &AppHandle, a: &Annotation) {
        self.emit_marker(app, &Marker::Annotate(a.clone()));
    }

    /// Mic level → HUD meter, throttled to ~16 Hz.
    pub fn level_emitter(&self, app: &AppHandle) -> impl Fn(f32) + Send + 'static {
        let app = app.clone();
        let last = Mutex::new(Instant::now());
        move |lvl| {
            let mut l = last.lock().unwrap();
            if l.elapsed().as_millis() >= 60 {
                *l = Instant::now();
                let _ = app.emit("luma://level", lvl);
            }
        }
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    pub fn set_pending(&self, p: Pending) {
        *self.pending.lock().unwrap() = Some(p);
    }

    /// Stop talking, clear the screen, cancel everything in flight. A paused
    /// task (`pending`) survives so the user's reply can resume it.
    pub fn interrupt(&self, app: &AppHandle) -> u64 {
        let e = self.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(s) = &self.speaker {
            s.stop();
        }
        self.speaking.store(false, Ordering::SeqCst);
        if let Some(l) = self.listening.lock().unwrap().take() {
            l.stt.abort();
            l.snapshot.abort();
        }
        *self.live_turn.lock().unwrap() = None;
        *self.watch.lock().unwrap() = None;
        let _ = app.emit("luma://clear", ());
        e
    }

    /// Cancel any paused task too (Stop button, Forget, Pause).
    pub fn cancel_all(&self, app: &AppHandle) -> u64 {
        *self.pending.lock().unwrap() = None;
        self.interrupt(app)
    }

    pub fn status(&self, app: &AppHandle, phase: Phase, message: Option<String>) {
        let display = *self.hud_display.lock().unwrap();
        let _ = app.emit("luma://status", StatusEvent { phase, message, display });
    }

    // ------------------------------------------------------------ hotkey

    /// Checks that must pass before listening. Returns false (and tells the
    /// user why) if LUMA can't run a turn right now.
    fn ready(&self, app: &AppHandle, prefs: &Prefs) -> bool {
        if prefs.paused {
            self.status(app, Phase::Paused, Some("LUMA is paused. Resume it from the menu bar.".into()));
            return false;
        }
        // Without an AssemblyAI key, macOS falls back to on-device speech.
        let required: &[Provider] = if cfg!(target_os = "macos") { &[Provider::Gemini] } else { &[Provider::Gemini, Provider::Assemblyai] };
        let missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|p| get_key(*p).is_none())
            .map(|p| p.name())
            .collect();
        if !missing.is_empty() {
            self.status(app, Phase::Error, Some(format!("Add your {} API key in LUMA settings.", missing.join(" and "))));
            crate::show_panel(app);
            return false;
        }
        if !screen::has_screen_permission() {
            let e = screen::request_screen_permission();
            self.status(app, Phase::Error, Some(e.to_string()));
            crate::show_panel(app);
            return false;
        }
        true
    }

    /// Open mic + STT (+ the voice stream, so it is ready when the answer
    /// starts). Returns false if the microphone is unavailable.
    fn start_listening(
        self: &Arc<Self>,
        app: &AppHandle,
        prefs: &Prefs,
        epoch: u64,
        snapshot: JoinHandle<Result<RawSnapshot>>,
        hands_free: bool,
    ) -> bool {
        let (mic_tx, mut mic_rx) = mpsc::channel::<Vec<u8>>(256);
        let (audio_tx, audio_rx) = mpsc::channel::<Vec<u8>>(256);
        let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<SttEvent>();
        // Tee: the cloud recognizer gets every chunk; a copy of the turn
        // (≤ 60 s, memory only) is kept for the on-device fallback.
        let recording: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let (rec_done_tx, rec_done_rx) = tokio::sync::oneshot::channel::<()>();
        {
            let recording = recording.clone();
            tauri::async_runtime::spawn(async move {
                let mut audio_tx = Some(audio_tx);
                while let Some(chunk) = mic_rx.recv().await {
                    {
                        let mut r = recording.lock().unwrap();
                        if r.len() < 60 * 2 * luma_net::STT_RATE as usize {
                            r.extend_from_slice(&chunk);
                        }
                    }
                    if let Some(tx) = &audio_tx {
                        if tx.send(chunk).await.is_err() {
                            audio_tx = None; // recognizer gave up; keep recording
                        }
                    }
                }
                let _ = rec_done_tx.send(());
            });
        }
        let mic = match start_mic(mic_tx, self.level_emitter(app)) {
            Ok(m) => m,
            Err(e) => {
                snapshot.abort();
                self.status(app, Phase::Error, Some(e.to_string()));
                return false;
            }
        };
        let key = get_key(Provider::Assemblyai).unwrap_or_default();
        let model = prefs.stt_model.clone();
        let app2 = app.clone();
        let language = prefs.tts_language.clone();
        let warm = self.warm_stt.lock().unwrap().take().filter(|c| c.hands_free() == hands_free && c.opened.elapsed() < WARM_STT_MAX_AGE);
        let stt = tauri::async_runtime::spawn(async move {
            let fwd = tokio::spawn(async move {
                while let Some(SttEvent::Partial(t)) = ev_rx.recv().await {
                    let _ = app2.emit("luma://transcript", &t);
                }
            });
            let r = match warm {
                Some(conn) => {
                    log::debug!("stt: using warm session ({} ms old)", conn.opened.elapsed().as_millis());
                    assemblyai::transcribe_on(conn, audio_rx, ev_tx).await
                }
                None if key.is_empty() => Err(anyhow!("no AssemblyAI key")),
                None => assemblyai::transcribe(&key, &model, audio_rx, ev_tx, hands_free).await,
            };
            fwd.abort();
            match r {
                // Push-to-talk only: the recording is complete once the key is released.
                Err(e) if !hands_free => {
                    log::warn!("cloud speech-to-text failed ({e:#}); trying on-device");
                    let _ = rec_done_rx.await;
                    let pcm = std::mem::take(&mut *recording.lock().unwrap());
                    let lang = language.clone();
                    match tauri::async_runtime::spawn_blocking(move || crate::local_stt::transcribe(&pcm, &lang)).await {
                        Ok(Ok(t)) => {
                            log::info!("on-device transcript used");
                            Ok(t)
                        }
                        Ok(Err(local)) => {
                            log::info!("on-device fallback unavailable: {local:#}");
                            Err(e)
                        }
                        Err(_) => Err(e),
                    }
                }
                other => other,
            }
        });
        let voice = {
            let this = self.clone();
            let prefs = prefs.clone();
            tauri::async_runtime::spawn(async move { this.open_voice(&prefs).await })
        };
        *self.listening.lock().unwrap() =
            Some(Listening { epoch, mic: Some(mic), stt, snapshot, voice: Some(voice), started: Instant::now() });
        true
    }

    // ------------------------------------------------------------ shortcut

    /// Shortcut down: stop whatever is happening and listen.
    pub fn on_press(self: &Arc<Self>, app: &AppHandle) {
        let t_press = Instant::now();
        let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
        let was_running_task = self.is_task_running();
        let epoch = self.interrupt(app);
        if was_running_task {
            *self.pending.lock().unwrap() = None;
            let _ = app.emit("luma://notice", "Stopped the task.");
        }
        if !self.ready(app, &prefs) {
            return;
        }
        let waiting = self.pending.lock().unwrap().is_some();
        if !waiting && crate::live_turn::use_live(self, &prefs) {
            if crate::live_turn::press(self, app, &prefs, epoch) {
                self.status(app, Phase::Listening, None);
            }
            return;
        }
        let snapshot = self.spawn_capture(app, &prefs, None);
        if !self.start_listening(app, &prefs, epoch, snapshot, false) {
            return;
        }
        self.status(app, Phase::Listening, waiting.then(|| "Listening for your answer…".to_string()));
        log::debug!("press → listening in {} ms", t_press.elapsed().as_millis());
    }

    /// Shortcut up: finish the transcript and answer (quick style: the model
    /// draws only when asked or when it clearly helps).
    pub fn on_release(self: &Arc<Self>, app: &AppHandle) {
        if crate::live_turn::release(self, app) {
            return;
        }
        let Some(mut l) = self.listening.lock().unwrap().take() else { return };
        // Closing the mic closes the audio channel, which forces the end of the turn.
        drop(l.mic.take());
        let held = l.started.elapsed();
        let t_release = Instant::now();
        let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
        self.status(app, Phase::Thinking, None);
        let this = self.clone();
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            let transcript = match l.stt.await {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => return this.fail(&app, l.epoch, &e),
                Err(_) => return, // aborted by barge-in
            };
            let stt_ms = t_release.elapsed().as_millis();
            log::info!("timing: transcript final {stt_ms} ms after release");
            let _ = app.emit("luma://timing", serde_json::json!({ "stt_ms": stt_ms }));
            if transcript.trim().is_empty() {
                l.snapshot.abort();
                if this.current(l.epoch) {
                    let msg = if held.as_millis() < 250 { "Hold the shortcut while you talk." } else { "I didn't catch that." };
                    let waiting = this.pending.lock().unwrap().is_some();
                    this.status(&app, if waiting { Phase::Waiting } else { Phase::Idle }, Some(msg.into()));
                }
                return;
            }
            this.respond(&app, l.epoch, transcript, l.snapshot, l.voice.take(), &prefs).await;
        });
    }

    /// A finished spoken request: a reply to a paused task or lesson, a local
    /// command, or a question for the model.
    async fn respond(
        self: &Arc<Self>,
        app: &AppHandle,
        epoch: u64,
        transcript: String,
        snapshot: JoinHandle<Result<RawSnapshot>>,
        mut voice: Option<JoinHandle<Option<VoiceConn>>>,
        prefs: &Prefs,
    ) {
        if !self.current(epoch) {
            snapshot.abort();
            return;
        }
        let _ = app.emit("luma://question", &transcript);
        if self.resume_pending(app, epoch, &transcript) {
            snapshot.abort();
            return;
        }
        if luma_core::lesson::is_continue(&transcript) && self.resume_lesson(app, epoch) {
            snapshot.abort();
            if let Some(v) = voice.take() {
                v.abort();
            }
            return;
        }
        if let Some(cmd) = local_command(&transcript) {
            snapshot.abort();
            return self.run_local(app, epoch, cmd, voice.take()).await;
        }
        let raw = match snapshot.await {
            Ok(s) => s,
            Err(_) => return,
        };
        // Asked to be shown something → precise pointing; otherwise the fast voice path.
        let style = if luma_core::prompt::wants_drawing(&transcript) && prefs.annotate != "never" {
            TurnStyle::pointing(prefs)
        } else {
            TurnStyle::quick(prefs)
        };
        log::info!("turn style: {} ({})", style.model, if style.draw == Draw::Always { "pointing" } else { "quick" });
        self.answer(app, epoch, transcript, raw, style, voice.take()).await;
    }

    /// After a turn has finished speaking: keep a speech-to-text session warm
    /// for a few seconds, or (opt-in) listen hands-free for a follow-up.
    fn after_turn(self: &Arc<Self>, app: &AppHandle, epoch: u64) {
        let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
        if prefs.paused || !self.current(epoch) || self.is_busy() || self.lesson_running.load(Ordering::SeqCst) {
            return;
        }
        if prefs.follow_up {
            let this = self.clone();
            let app = app.clone();
            tauri::async_runtime::spawn(async move { this.listen_follow_up(&app, epoch).await });
            return;
        }
        if prefs.warm_stt {
            let this = self.clone();
            let key = get_key(Provider::Assemblyai);
            tauri::async_runtime::spawn(async move {
                let Some(key) = key else { return };
                match assemblyai::connect(&key, &prefs.stt_model, false).await {
                    Ok(c) if this.current(epoch) => *this.warm_stt.lock().unwrap() = Some(c),
                    Ok(_) => {}
                    Err(e) => log::debug!("warm stt: {e}"),
                }
            });
        }
    }

    /// A task paused for the user's yes/no or an answer: in follow-up mode,
    /// listen for it hands-free once the question has been spoken.
    pub fn listen_for_reply_when_quiet(self: &Arc<Self>, app: &AppHandle, epoch: u64) {
        if !app.state::<AppState>().prefs.lock().unwrap().follow_up {
            return;
        }
        let this = self.clone();
        let app = app.clone();
        let go = move || {
            let this2 = this.clone();
            let app2 = app.clone();
            tauri::async_runtime::spawn(async move { this2.listen_follow_up(&app2, epoch).await });
        };
        match &self.speaker {
            Some(s) => s.send(SpeakerCmd::Callback(Box::new(move || go()))),
            None => go(),
        }
    }

    /// Opt-in conversation mode: once LUMA has finished talking, listen
    /// hands-free for a follow-up ("yes", "and that one?") for a few seconds.
    /// The mic opens only after LUMA stops speaking, so it never hears itself.
    async fn listen_follow_up(self: &Arc<Self>, app: &AppHandle, epoch: u64) {
        let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
        if !self.current(epoch) || !self.ready(app, &prefs) {
            return;
        }
        let snapshot = self.spawn_capture(app, &prefs, None);
        if !self.start_listening(app, &prefs, epoch, snapshot, true) {
            return;
        }
        let waiting = self.pending.lock().unwrap().is_some();
        self.status(app, Phase::Listening, Some(if waiting { "Listening for your answer…" } else { "Anything else? I'm listening…" }.into()));
        let stt = {
            let mut guard = self.listening.lock().unwrap();
            match guard.as_mut() {
                Some(l) if l.epoch == epoch => std::mem::replace(&mut l.stt, tauri::async_runtime::spawn(async { Ok(String::new()) })),
                _ => return,
            }
        };
        let transcript = match stt.await {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => return self.fail(app, epoch, &e),
            Err(_) => return,
        };
        let Some(mut l) = self.listening.lock().unwrap().take().filter(|l| l.epoch == epoch) else { return };
        drop(l.mic.take());
        if transcript.trim().is_empty() {
            l.snapshot.abort();
            if let Some(v) = l.voice.take() {
                v.abort();
            }
            if self.current(epoch) {
                let waiting = self.pending.lock().unwrap().is_some();
                self.status(app, if waiting { Phase::Waiting } else { Phase::Idle }, None);
            }
            return;
        }
        self.status(app, Phase::Thinking, None);
        self.respond(app, epoch, transcript, l.snapshot, l.voice.take(), &prefs).await;
    }

    // ------------------------------------------------------------ gesture

    /// Long-press on the trackpad/mouse (held still ≥ long_press_ms): point
    /// at something and ask about it hands-free, with drawing.
    pub fn on_gesture(self: &Arc<Self>, app: &AppHandle, at: luma_core::geometry::Point) {
        let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
        if self.listening.lock().unwrap().is_some() || self.is_task_running() {
            return;
        }
        let epoch = self.interrupt(app);
        if !self.ready(app, &prefs) {
            return;
        }
        let _ = app.emit("luma://mode", "annotate");
        let snapshot = self.spawn_capture(app, &prefs, Some(at));
        if !self.start_listening(app, &prefs, epoch, snapshot, true) {
            return;
        }
        self.status(app, Phase::Listening, Some("Ask about this, or stay quiet and I'll explain it".into()));
        let this = self.clone();
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            // hands-free STT ends on its own; take the listening state back
            let stt = {
                let mut guard = this.listening.lock().unwrap();
                match guard.as_mut() {
                    Some(l) if l.epoch == epoch => std::mem::replace(&mut l.stt, tauri::async_runtime::spawn(async { Ok(String::new()) })),
                    _ => return,
                }
            };
            let transcript = match stt.await {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => return this.fail(&app, epoch, &e),
                Err(_) => return,
            };
            let Some(mut l) = this.listening.lock().unwrap().take().filter(|l| l.epoch == epoch) else { return };
            drop(l.mic.take());
            if !this.current(epoch) {
                return;
            }
            let question = if transcript.trim().is_empty() {
                "What is this? Explain what I'm pointing at.".to_string()
            } else {
                transcript
            };
            let _ = app.emit("luma://question", &question);
            this.status(&app, Phase::Thinking, None);
            let raw = match l.snapshot.await {
                Ok(s) => s,
                Err(_) => return,
            };
            this.answer(&app, epoch, question, raw, TurnStyle::pointing(&prefs), l.voice.take()).await;
        });
    }

    /// Returns true if the reply was consumed by a paused task.
    fn resume_pending(self: &Arc<Self>, app: &AppHandle, epoch: u64, reply: &str) -> bool {
        let Some(p) = self.pending.lock().unwrap().take() else { return false };
        match p {
            Pending::Approval { task, action } => match yes_no(reply) {
                YesNo::Yes => {
                    tauri::async_runtime::spawn(agent::run(self.clone(), app.clone(), epoch, task, Some(action)));
                    true
                }
                YesNo::No => {
                    self.say(app, epoch, "Okay, I won't do that. I've stopped the task.");
                    self.finish_when_quiet(app, epoch);
                    true
                }
                // Something else entirely: drop the task, treat it as a new request.
                YesNo::Unclear => false,
            },
            Pending::Answer { mut task } => {
                task.history.push(format!("the user said: \"{reply}\""));
                task.last_changed = None;
                tauri::async_runtime::spawn(agent::run(self.clone(), app.clone(), epoch, task, None));
                true
            }
        }
    }

    /// Typed question (panel input): pointing style, since there is no voice
    /// gesture to choose with.
    pub fn ask_text(self: &Arc<Self>, app: &AppHandle, question: String) {
        let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
        let epoch = self.interrupt(app);
        if !self.ready(app, &prefs) {
            return;
        }
        let _ = app.emit("luma://question", &question);
        if self.resume_pending(app, epoch, &question) {
            return;
        }
        if luma_core::lesson::is_continue(&question) && self.resume_lesson(app, epoch) {
            return;
        }
        if let Some(cmd) = local_command(&question) {
            let this = self.clone();
            let app = app.clone();
            tauri::async_runtime::spawn(async move { this.run_local(&app, epoch, cmd, None).await });
            return;
        }
        let snapshot = self.spawn_capture(app, &prefs, None);
        self.status(app, Phase::Thinking, None);
        let this = self.clone();
        let app = app.clone();
        tauri::async_runtime::spawn(async move {
            if let Ok(raw) = snapshot.await {
                this.answer(&app, epoch, question, raw, TurnStyle::pointing(&prefs), None).await;
            }
        });
    }

    pub fn spawn_capture(&self, app: &AppHandle, prefs: &Prefs, at: Option<luma_core::geometry::Point>) -> JoinHandle<Result<RawSnapshot>> {
        let app = app.clone();
        let prefs = prefs.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let displays = screen::displays()?;
            let pointer = at.or_else(|| screen::pointer(&app, &displays));
            if let Some(p) = pointer.and_then(|p| luma_core::geometry::display_at(&displays, p)) {
                let st = app.state::<AppState>();
                *st.companion.hud_display.lock().unwrap() = Some(p.index);
            }
            crate::sync_overlays(&app, &displays);
            if let Some(w) = screen::active_window() {
                if prefs.is_excluded(&w.app) {
                    return Err(anyhow!(Excluded(w.app)));
                }
            }
            screen::capture(displays, pointer, None)
        })
    }

    fn fail(&self, app: &AppHandle, epoch: u64, e: &anyhow::Error) {
        if !self.current(epoch) {
            return;
        }
        log::warn!("turn failed: {e:#}");
        self.status(app, Phase::Error, Some(assemblyai::redact(&format!("{e}"))));
    }

    /// The accurate ("pointing") model, used by the agent.
    pub fn gemini_for(&self, prefs: &Prefs, media_resolution: &'static str) -> Gemini {
        Gemini {
            client: self.http.clone(),
            api_key: get_key(Provider::Gemini).unwrap_or_default(),
            model: prefs.gemini_model.clone(),
            thinking_level: prefs.thinking_level.clone(),
            media_resolution,
        }
    }

    fn gemini_for_style(&self, _prefs: &Prefs, style: &TurnStyle) -> Gemini {
        Gemini {
            client: self.http.clone(),
            api_key: get_key(Provider::Gemini).unwrap_or_default(),
            model: style.model.clone(),
            thinking_level: style.thinking.clone(),
            media_resolution: style.media,
        }
    }

    fn tts(&self, prefs: &Prefs) -> Option<SarvamTts> {
        get_key(Provider::Sarvam).filter(|_| self.speaker.is_some()).map(|k| SarvamTts {
            client: self.http.clone(),
            api_key: k,
            speaker: prefs.tts_speaker.clone(),
            language: prefs.tts_language.clone(),
            pace: prefs.tts_pace,
        })
    }

    /// Re-ask on a tight native-resolution crop for a small mark; swap in
    /// the better box (now, if the mark is already on screen, or when it is
    /// shown).
    /// Start a fresh set of marks for `epoch` (refined geometry is per turn).
    pub fn begin_marks(&self, epoch: u64) {
        *self.refined.lock().unwrap() = RefineBook { epoch, ..Default::default() };
    }

    /// Make a mark exact in the background, never holding up speech: snap it
    /// to an accessibility frame or OCR text lines (cheap, on device), and if
    /// that finds nothing and the mark is small, re-ask on a close-up crop.
    /// The better box replaces the mark whether or not it is on screen yet.
    pub fn spawn_improve(self: &Arc<Self>, app: &AppHandle, epoch: u64, raw: Arc<RawSnapshot>, a: Annotation, prefs: &Prefs) {
        use luma_core::annotation::ShapeKind;
        use luma_core::snap::SnapKind;
        let Some((display, target)) = a.target() else { return };
        let Some(id) = a.id().map(str::to_string) else { return };
        let label = match &a {
            Annotation::Shape { label, .. } | Annotation::Point { label, .. } | Annotation::Step { label, .. } => label.clone(),
            _ => None,
        };
        let kind = match &a {
            Annotation::Shape { kind: ShapeKind::Highlight | ShapeKind::Underline, .. } => SnapKind::Text,
            Annotation::Point { .. } => SnapKind::Point,
            _ => SnapKind::Area,
        };
        let snap = prefs.snap_to_elements;
        let refine = prefs.refine_small && kind != SnapKind::Text && luma_core::refine::needs_refine(&target);
        if !snap && !refine {
            return;
        }
        let this = self.clone();
        let app = app.clone();
        let gemini = Gemini {
            client: self.http.clone(),
            api_key: get_key(Provider::Gemini).unwrap_or_default(),
            model: prefs.fast_model.clone(),
            thinking_level: "minimal".into(),
            media_resolution: "MEDIA_RESOLUTION_HIGH",
        };
        tokio::spawn(async move {
            let t = Instant::now();
            if snap {
                let raw2 = raw.clone();
                let label2 = label.clone();
                let snapped = tauri::async_runtime::spawn_blocking(move || {
                    let mut f = element_snapper(raw2.displays.clone(), raw2.clone());
                    f(&luma_core::geometry::DisplayRect { display_index: display, rect: target }, label2.as_deref(), kind)
                })
                .await
                .ok()
                .flatten();
                if let Some(r) = snapped {
                    log::debug!("snapped {id} in {} ms", t.elapsed().as_millis());
                    return this.apply_better(&app, epoch, &id, a.with_rect(r));
                }
            }
            if !refine || !this.current(epoch) {
                return;
            }
            let raw2 = raw.clone();
            let prepared = tauri::async_runtime::spawn_blocking(move || {
                let cap = luma_core::geometry::Capture {
                    display_index: raw2.display.index,
                    width_px: raw2.full.width(),
                    height_px: raw2.full.height(),
                };
                let crop = luma_core::refine::refine_crop(&target, &raw2.display, cap);
                luma_net::vision::encode(&raw2.full, &crop).map(|jpeg| (crop, jpeg))
            })
            .await;
            let Ok(Ok((crop, jpeg))) = prepared else { return };
            let what = luma_core::refine::describe_target(&id, label.as_deref());
            let body = gemini.build_body(&luma_core::refine::refine_prompt(&what), &[], "", &[ImagePart { jpeg: &jpeg }], "Find it.");
            let reply = match gemini.complete(&body).await {
                Ok(r) => r,
                Err(e) => return log::debug!("refine failed: {e}"),
            };
            if !this.current(epoch) {
                return;
            }
            let Some(r) = luma_core::refine::parse_refined(&reply, &crop, &raw.display, &target) else {
                return log::debug!("refine kept {id} ({} ms): {reply:?}", t.elapsed().as_millis());
            };
            log::debug!("refined {id} in {} ms: {target:?} → {r:?}", t.elapsed().as_millis());
            this.apply_better(&app, epoch, &id, a.with_rect(r));
        });
    }

    /// Swap in better geometry for mark `id`: now if it is on screen, or
    /// when it is shown; and in the session's memory.
    fn apply_better(&self, app: &AppHandle, epoch: u64, id: &str, better: Annotation) {
        let shown = {
            let mut book = self.refined.lock().unwrap();
            if book.epoch != epoch || !self.current(epoch) {
                return;
            }
            book.better.insert(id.to_string(), better.clone());
            book.shown.contains(id)
        };
        if shown {
            self.emit_marker(app, &Marker::Annotate(better.clone()));
        }
        if let Some((d, rect)) = better.target() {
            let st = app.state::<AppState>();
            st.session.lock().unwrap().update_item(id, luma_core::geometry::DisplayRect { display_index: d, rect });
        }
    }

    // ------------------------------------------------------------ answer

    async fn answer(
        self: &Arc<Self>,
        app: &AppHandle,
        epoch: u64,
        question: String,
        raw: Result<RawSnapshot>,
        style: TurnStyle,
        voice: Option<JoinHandle<Option<VoiceConn>>>,
    ) {
        let raw = match raw {
            Ok(s) => s,
            Err(e) => {
                if let Some(x) = e.downcast_ref::<Excluded>() {
                    self.status(app, Phase::Idle, Some(x.to_string()));
                } else {
                    self.fail(app, epoch, &e);
                }
                return;
            }
        };
        let st = app.state::<AppState>();
        let prefs = st.prefs.lock().unwrap().clone();
        let opts = CaptureOptions { max_edge: style.max_edge, closeup: style.closeup };
        let raw = Arc::new(raw);
        let raw2 = raw.clone();
        let snap: Snapshot = match tauri::async_runtime::spawn_blocking(move || raw2.encode(&opts)).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => return self.fail(app, epoch, &e),
            Err(_) => return,
        };
        let context_key = snap.context_key();
        let draw = style.draw != Draw::Never;

        let (body, fallback, prior_items, turn_no) = {
            let mut session = st.session.lock().unwrap();
            session.observe_level_request(&question);
            let marked = if draw { session.describe_items(&context_key, &snap.images[0].sent, &snap.displays) } else { String::new() };
            let focused = snap.focus.as_ref().and_then(|f| luma_core::prompt::describe_focus(&f.role, &f.name, f.secure));
            let lesson_line = self.lesson().map(|l| format!("{} — {}", l.goal, l.status_line()));
            let ctx = context_block(&TurnContext {
                app: Some(snap.window.app.as_str()).filter(|s| !s.is_empty()),
                window_title: Some(snap.window.title.as_str()),
                pointer: snap.pointer_norm,
                pointer_closeup: snap.pointer_norm_closeup,
                has_closeup: snap.images.len() > 1,
                display_count: snap.displays.len(),
                level: session.level,
                marked_items: &marked,
                focused: focused.as_deref(),
                selection: snap.focus.as_ref().and_then(|f| f.selection.as_deref()),
                lesson: lesson_line.as_deref(),
            });
            let history: Vec<HistoryTurn> =
                session.turns().map(|t| HistoryTurn { user: &t.user, assistant: &t.assistant }).collect();
            let images: Vec<ImagePart> = snap.images.iter().map(|i| ImagePart { jpeg: &i.jpeg }).collect();
            let system = system_prompt(style.draw, prefs.autopilot);
            let body = self.gemini_for_style(&prefs, &style).build_body(&system, &history, &ctx, &images, &question);
            // the fast model, in case the accurate one deliberates for long
            let fallback = (style.model != prefs.fast_model).then(|| {
                let fast = TurnStyle { model: prefs.fast_model.clone(), thinking: prefs.fast_thinking.clone(), ..style.clone() };
                let g = self.gemini_for_style(&prefs, &fast);
                let b = g.build_body(&system, &history, &ctx, &images, &question);
                (g, b)
            });
            (body, fallback, session.items_for(&context_key), session.next_turn_number())
        };

        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let gemini = self.gemini_for_style(&prefs, &style);
        let t_request = Instant::now();
        let llm = gemini.hedged(body, fallback, std::time::Duration::from_millis(HEDGE_AFTER_MS), tx);
        // A long silence feels broken: if nothing has arrived after a few
        // seconds (the model is planning a drawing), say so.
        let answer_started = Arc::new(AtomicBool::new(false));
        {
            let this = self.clone();
            let app = app.clone();
            let started = answer_started.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(SILENCE_FILLER_MS)).await;
                if started.load(Ordering::SeqCst) || !this.current(epoch) {
                    return;
                }
                this.status(&app, Phase::Thinking, Some("Working it out…".into()));
                this.speak_unless(&app, epoch, "One moment, let me work that out.", &started).await;
            });
        }

        let voice = match voice {
            Some(v) => v.await.ok().flatten(),
            None => self.open_voice(&prefs).await,
        };
        if get_key(Provider::Sarvam).is_none() {
            let _ = app.emit("luma://notice", "No Sarvam key: answers are shown as captions only.");
        }
        let (out, output_done) = self.start_output(app, epoch, voice, &prefs, t_request);
        let images: Vec<_> = snap.images.iter().map(|i| i.sent).collect();
        let mut resolver = Resolver::new(&snap.displays, &images, prior_items, turn_no);
        // No snapper here: snapping runs in the background (spawn_improve),
        // so a slow app's accessibility tree never holds up the answer.
        let mut parser = MarkupParser::new();
        let mut chunker = SpeechChunker::new();
        let mut spoken = String::new();
        let mut task_goal: Option<String> = None;
        let mut lesson_goal: Option<String> = None;
        let mut first_token: Option<u128> = None;
        let draw = style.draw != Draw::Never;
        let mut replay: Vec<Out> = Vec::new();
        let mut emit = |o: Out| {
            replay.push(o.clone());
            let _ = out.send(o);
        };
        self.begin_marks(epoch);
        let mut improves = 0usize;
        let mut after_mark = |a: &Annotation, emit: &mut dyn FnMut(Out)| {
            if let Annotation::Zoom { display, rect } = a {
                match luma_net::vision::magnifier_data_url(&raw.full, &raw.display, rect) {
                    Ok(src) => emit(Out::Mark(Marker::Magnify { display: *display, rect: *rect, src })),
                    Err(e) => log::debug!("no magnifier: {e}"),
                }
            }
            if a.target().is_some() && a.id().is_some() && improves < MAX_IMPROVES {
                improves += 1;
                self.spawn_improve(app, epoch, raw.clone(), a.clone(), &prefs);
            }
        };

        let mut handle_segments = |segs: Vec<Segment>, task_goal: &mut Option<String>, lesson_goal: &mut Option<String>| {
            for s in segs {
                match s {
                    Segment::Text(t) => {
                        spoken.push_str(&t);
                        for u in chunker.push(&t) {
                            emit(Out::Mark(Marker::Caption(u.clone())));
                            emit(Out::Speech(u));
                        }
                    }
                    Segment::Tag(tag) if tag.name == "task" => {
                        if let Some(g) = tag.attr("goal").map(str::trim).filter(|g| !g.is_empty()) {
                            task_goal.get_or_insert_with(|| g.to_string());
                        }
                    }
                    Segment::Tag(tag) if tag.name == "lesson" => {
                        if let Some(g) = tag.attr("goal").map(str::trim).filter(|g| !g.is_empty()) {
                            lesson_goal.get_or_insert_with(|| g.to_string());
                        }
                    }
                    Segment::Tag(_) if !draw => {}
                    Segment::Tag(tag) => match resolver.resolve(&tag) {
                        Ok(a) => {
                            // freehand traces land exactly on the drawn line, and LUMA's
                            // own diagram goes where it won't cover content (a few ms, on device)
                            let a = if a.display() == Some(raw.display.index) {
                                let a = luma_net::vision::place_board(a, &mut resolver, &raw.full, &raw.display);
                                luma_net::vision::snap_sketch_to_ink(&a, &raw.full, &raw.display)
                            } else {
                                a
                            };
                            emit(Out::Mark(Marker::Annotate(a.clone())));
                            after_mark(&a, &mut emit);
                            for ready in resolver.take_ready() {
                                emit(Out::Mark(Marker::Annotate(ready)));
                            }
                        }
                        Err(e) => log::debug!("dropped tag <{}>: {}", e.tag, e.reason),
                    },
                }
            }
        };

        while let Some(delta) = rx.recv().await {
            if !self.current(epoch) {
                llm.abort();
                return;
            }
            first_token.get_or_insert_with(|| t_request.elapsed().as_millis());
            answer_started.store(true, Ordering::SeqCst);
            let segs = parser.push(&delta);
            handle_segments(segs, &mut task_goal, &mut lesson_goal);
        }
        let segs = parser.finish();
        handle_segments(segs, &mut task_goal, &mut lesson_goal);
        drop(handle_segments);
        drop(after_mark);
        if let Some(u) = chunker.finish() {
            emit(Out::Mark(Marker::Caption(u.clone())));
            emit(Out::Speech(u));
        }
        drop(emit);
        drop(out); // lets the output consumer finish once everything is queued
        if replay.iter().any(|o| matches!(o, Out::Speech(_))) {
            *self.last_reply.lock().unwrap() = replay;
        }

        let answered_by = match llm.await {
            Ok(Ok(model)) => model,
            Ok(Err(e)) => return self.fail(app, epoch, &e),
            Err(_) => return,
        };
        if !self.current(epoch) {
            return;
        }
        let ft = first_token.unwrap_or(0);
        log::info!("timing: first token {ft} ms ({answered_by} / {})", style.media);
        let _ = app.emit("luma://timing", serde_json::json!({ "first_token_ms": ft, "model": style.model }));

        // ---- remember this turn (text + marked items only)
        for (id, a) in self.refined.lock().unwrap().better.iter() {
            if let Some((d, r)) = a.target() {
                resolver.update_item(id, luma_core::geometry::DisplayRect { display_index: d, rect: r });
            }
        }
        let items: Vec<_> = resolver.items.values().filter(|m| m.turn == turn_no).cloned().collect();
        let assistant = luma_core::speech::clean(&spoken).unwrap_or_default();
        st.session.lock().unwrap().record(
            Turn { user: question.clone(), assistant: assistant.clone(), context_key },
            items,
        );
        if !assistant.is_empty() {
            let _ = app.emit("luma://answer", &assistant);
        }

        // ---- hand off to the agent, or go idle when the speech has played
        let this = self.clone();
        let app2 = app.clone();
        let display = snap.display.index;
        let window_app = snap.window.app.clone();
        let window_context = format!("{} {}", snap.window.app, snap.window.title);
        tokio::spawn(async move {
            let _ = output_done.await;
            if !this.current(epoch) {
                return;
            }
            // Guide mode (autopilot off): a task the model started anyway
            // becomes a lesson, so the user does the clicking.
            let (lesson_goal, task_goal) = match (lesson_goal, task_goal) {
                (None, Some(goal)) if !prefs.autopilot => (Some(goal), None),
                goals => goals,
            };
            if let Some(goal) = lesson_goal {
                let mut l = luma_core::lesson::Lesson::new(goal);
                l.app = window_app;
                crate::lesson::persist(&app2, Some(&l));
                this.set_lesson(Some(l));
                crate::lesson::run(this.clone(), app2.clone(), epoch, Some(display)).await;
                return;
            }
            // Prompt-injection guard: specific values in the goal must come
            // from the user, not from text on the screen.
            let task_goal = match task_goal {
                Some(goal) => match luma_core::action::ungrounded_value(&goal, &question, &window_context) {
                    Some(v) => {
                        log::warn!("task refused: {v:?} is not in the request");
                        this.speak(&app2, epoch, &format!("I didn't start that, because \"{v}\" came from the screen, not from you. If you meant it, tell me yourself.")).await;
                        this.finish_when_quiet(&app2, epoch);
                        return;
                    }
                    None => Some(goal),
                },
                None => None,
            };
            match task_goal {
                Some(goal) if prefs.autopilot => {
                    agent::run(this.clone(), app2.clone(), epoch, TaskState::new(goal, Some(display)), None).await;
                }
                _ if this.lesson().is_some() => {
                    // A question in the middle of a lesson: carry on watching.
                    crate::lesson::run(this.clone(), app2.clone(), epoch, Some(display)).await;
                }
                _ => {
                    if assistant.is_empty() {
                        this.status(&app2, Phase::Idle, Some("I don't have an answer for that.".into()));
                    } else {
                        this.finish_when_quiet(&app2, epoch);
                    }
                }
            }
        });
    }

    /// "Never mind" / "repeat that": answered locally, no screenshot or model call.
    async fn run_local(
        self: &Arc<Self>,
        app: &AppHandle,
        epoch: u64,
        cmd: LocalCommand,
        voice: Option<JoinHandle<Option<VoiceConn>>>,
    ) {
        log::info!("local command: {cmd:?}");
        match cmd {
            LocalCommand::Cancel => {
                if let Some(v) = voice {
                    v.abort();
                }
                self.emit_marker(app, &Marker::Annotate(Annotation::Clear { id: None }));
                let had_lesson = self.lesson.lock().unwrap().take().is_some();
                crate::lesson::persist(app, None);
                self.status(app, Phase::Idle, Some(if had_lesson { "Okay, I've stopped the lesson." } else { "Okay." }.into()));
            }
            LocalCommand::WhatChanged => {
                if let Some(v) = voice {
                    v.abort();
                }
                let log = self.last_task.lock().unwrap().clone();
                let line = match log {
                    None => "I haven't changed anything in this session.".to_string(),
                    Some(t) => {
                        let changes: Vec<&String> =
                            t.actions.iter().filter(|a| luma_core::action::history_line_modifies(a)).collect();
                        if changes.is_empty() {
                            format!("For \"{}\" I only looked around and opened pages; I didn't change anything.", t.goal)
                        } else {
                            let list: Vec<String> = changes.iter().map(|a| a.split(" (").next().unwrap_or(a).to_string()).collect();
                            format!("For \"{}\" in {}, I {}. The result: {}.", t.goal, if t.app.is_empty() { "the app" } else { &t.app }, list.join(", then "), t.outcome)
                        }
                    }
                };
                let _ = app.emit("luma://answer", &line);
                self.speak(app, epoch, &line).await;
                self.finish_when_quiet(app, epoch);
            }
            LocalCommand::Undo => {
                if let Some(v) = voice {
                    v.abort();
                }
                let line = self.undo_last().await;
                let _ = app.emit("luma://answer", &line);
                self.speak(app, epoch, &line).await;
                self.finish_when_quiet(app, epoch);
            }
            LocalCommand::Repeat => {
                let items = self.last_reply.lock().unwrap().clone();
                if items.is_empty() {
                    if let Some(v) = voice {
                        v.abort();
                    }
                    self.status(app, Phase::Idle, Some("Nothing to repeat yet.".into()));
                    return;
                }
                let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
                let voice = match voice {
                    Some(v) => v.await.ok().flatten(),
                    None => self.open_voice(&prefs).await,
                };
                let (out, done) = self.start_output(app, epoch, voice, &prefs, Instant::now());
                let _ = out.send(Out::Mark(Marker::Annotate(Annotation::Clear { id: None })));
                for o in items {
                    let _ = out.send(o);
                }
                drop(out);
                let _ = done.await;
                if self.current(epoch) {
                    self.finish_when_quiet(app, epoch);
                }
            }
        }
    }

    /// "Undo that": one app-level undo in the app the last task changed,
    /// checked against the screen, and reported honestly.
    async fn undo_last(self: &Arc<Self>) -> String {
        let Some(t) = self.last_task.lock().unwrap().clone() else {
            return "There's nothing of mine to undo.".into();
        };
        let Some(last) = t.actions.iter().rev().find(|a| luma_core::action::history_line_modifies(a)).cloned() else {
            return "My last task didn't change anything, so there's nothing to undo.".into();
        };
        let last = last.split(" (").next().unwrap_or(&last).to_string();
        let front = screen::active_window().map(|w| w.app).unwrap_or_default();
        if !t.app.is_empty() && front != t.app {
            return format!("Switch back to {} and ask me again; I only undo inside the app I changed.", t.app);
        }
        if !crate::input::has_control_permission() {
            let _ = crate::input::request_control_permission();
            return "I need Accessibility access to press undo for you. I've opened the settings.".into();
        }
        let r = tauri::async_runtime::spawn_blocking(move || -> Result<f64> {
            let displays = screen::displays()?;
            let idx = screen::pointer_native()
                .and_then(|p| luma_core::geometry::display_at(&displays, p))
                .map(|d| d.index)
                .unwrap_or(0);
            let before = luma_net::vision::thumbnail(&screen::capture_display(idx)?);
            crate::input::press_keys(&["cmd".into(), "z".into()])?;
            std::thread::sleep(std::time::Duration::from_millis(700));
            let after = luma_net::vision::thumbnail(&screen::capture_display(idx)?);
            Ok(luma_net::vision::changed_fraction(&before, &after))
        })
        .await;
        match r {
            Ok(Ok(changed)) if changed > 0.004 => format!(
                "I pressed undo once in {}, which should reverse \"{last}\". Undo goes one step at a time, so check it looks right, and ask again to undo more.",
                if t.app.is_empty() { "the app" } else { &t.app }
            ),
            Ok(Ok(_)) => format!(
                "I pressed undo, but nothing on screen changed, so this app may not be able to undo that. You'll need to change it back by hand: I {last}."
            ),
            _ => format!("I couldn't press undo. To reverse it by hand: I {last}."),
        }
    }

    /// Open the streaming voice for a turn (None: no key, no speaker, or offline).
    pub async fn open_voice(&self, prefs: &Prefs) -> Option<VoiceConn> {
        let key = get_key(Provider::Sarvam)?;
        self.speaker.as_ref()?;
        let cfg = TtsConfig {
            api_key: key,
            speaker: prefs.tts_speaker.clone(),
            language: prefs.tts_language.clone(),
            pace: prefs.tts_pace,
        };
        match sarvam_ws::open(&cfg).await {
            Ok((stream, events)) => Some(VoiceConn { stream, events }),
            Err(e) => {
                log::warn!("streaming voice unavailable, falling back to REST: {e}");
                None
            }
        }
    }

    /// The single ordered output of a turn: speech (streamed into the
    /// speaker as it is synthesized) and markers (annotations, captions)
    /// queued between the audio, so each fires exactly when its sentence
    /// starts. Returns when everything has been queued on the speaker.
    fn start_output(
        self: &Arc<Self>,
        app: &AppHandle,
        epoch: u64,
        mut voice: Option<VoiceConn>,
        prefs: &Prefs,
        t_request: Instant,
    ) -> (mpsc::UnboundedSender<Out>, tokio::task::JoinHandle<()>) {
        let (tx, mut rx) = mpsc::unbounded_channel::<Out>();
        let this = self.clone();
        let app = app.clone();
        let rest = self.tts(prefs);
        let done = tokio::spawn(async move {
            let mut first_audio = true;
            let mut audio_logged = false;
            while let Some(item) = rx.recv().await {
                if !this.current(epoch) {
                    return;
                }
                match item {
                    Out::Mark(m) => match &this.speaker {
                        Some(s) => {
                            let this2 = this.clone();
                            let app2 = app.clone();
                            s.send(SpeakerCmd::Callback(Box::new(move || {
                                if this2.current(epoch) {
                                    this2.emit_marker(&app2, &m);
                                }
                            })));
                        }
                        None => this.emit_marker(&app, &m),
                    },
                    Out::Speech(text) => {
                        if first_audio {
                            first_audio = false;
                            this.speaking.store(true, Ordering::SeqCst);
                            this.status(&app, Phase::Speaking, None);
                        }
                        let Some(speaker) = this.speaker.clone() else { continue };
                        let mut streamed = false;
                        if let Some(v) = voice.as_mut() {
                            if v.stream.say(&text) {
                                streamed = true;
                                loop {
                                    match v.events.recv().await {
                                        Some(TtsEvent::Audio(pcm)) => {
                                            if !this.current(epoch) {
                                                return;
                                            }
                                            if !audio_logged {
                                                audio_logged = true;
                                                let ms = t_request.elapsed().as_millis();
                                                log::info!("timing: first audio {ms} ms after request");
                                                let _ = app.emit("luma://timing", serde_json::json!({ "first_audio_ms": ms }));
                                            }
                                            speaker.send(SpeakerCmd::Pcm(pcm, sarvam_ws::SAMPLE_RATE));
                                        }
                                        Some(TtsEvent::End) => break,
                                        Some(TtsEvent::Error(e)) => {
                                            log::warn!("streaming voice error: {e}");
                                            streamed = false;
                                            voice = None;
                                            break;
                                        }
                                        None => {
                                            streamed = false;
                                            voice = None;
                                            break;
                                        }
                                    }
                                }
                            } else {
                                voice = None;
                            }
                        }
                        if !streamed {
                            if let Some(r) = &rest {
                                match r.synthesize(&text).await {
                                    Ok(wav) if this.current(epoch) => speaker.send(SpeakerCmd::Play(wav)),
                                    Ok(_) => return,
                                    Err(e) => log::warn!("tts failed: {e}"),
                                }
                            }
                        }
                    }
                }
            }
        });
        (tx, done)
    }

    /// Show `phase` once everything queued on the speaker has played.
    pub fn status_when_quiet(self: &Arc<Self>, app: &AppHandle, epoch: u64, phase: Phase, message: Option<String>) {
        match &self.speaker {
            // An empty queue runs the callback at once.
            Some(s) => {
                let this = self.clone();
                let app = app.clone();
                s.send(SpeakerCmd::Callback(Box::new(move || {
                    if this.current(epoch) {
                        this.speaking.store(false, Ordering::SeqCst);
                        this.status(&app, phase, message.clone());
                    }
                })));
            }
            _ => {
                self.speaking.store(false, Ordering::SeqCst);
                self.status(app, phase, message);
            }
        }
    }

    /// Go idle once everything queued on the speaker has played.
    pub fn finish_when_quiet(self: &Arc<Self>, app: &AppHandle, epoch: u64) {
        match &self.speaker {
            // An empty queue runs the callback at once.
            Some(s) => {
                let this = self.clone();
                let app = app.clone();
                s.send(SpeakerCmd::Callback(Box::new(move || {
                    if this.current(epoch) {
                        this.speaking.store(false, Ordering::SeqCst);
                        this.status(&app, Phase::Idle, None);
                        this.after_turn(&app, epoch);
                    }
                })));
            }
            _ => {
                self.speaking.store(false, Ordering::SeqCst);
                self.status(app, Phase::Idle, None);
                self.after_turn(app, epoch);
            }
        }
    }

    /// Speak a filler line, unless `cancel` became true while it was being
    /// synthesized (the real answer started): never plays inside an answer.
    async fn speak_unless(self: &Arc<Self>, app: &AppHandle, epoch: u64, text: &str, cancel: &AtomicBool) {
        let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
        let Some(tts) = self.tts(&prefs) else { return };
        let Ok(audio) = tts.synthesize(text).await else { return };
        if cancel.load(Ordering::SeqCst) || !self.current(epoch) || self.is_speaking() {
            return;
        }
        if let Some(s) = &self.speaker {
            s.send(SpeakerCmd::Play(audio));
        }
    }

    /// Speak one line outside the streaming pipeline, without waiting.
    pub fn say(self: &Arc<Self>, app: &AppHandle, epoch: u64, text: &str) {
        let this = self.clone();
        let app = app.clone();
        let text = text.to_string();
        tauri::async_runtime::spawn(async move { this.speak(&app, epoch, &text).await });
    }

    /// Speak one line; returns once its audio is queued on the speaker, so
    /// anything queued afterwards plays after it.
    pub async fn speak(self: &Arc<Self>, app: &AppHandle, epoch: u64, text: &str) {
        let text = text.trim().to_string();
        if text.is_empty() || !self.current(epoch) {
            return;
        }
        let prefs = app.state::<AppState>().prefs.lock().unwrap().clone();
        let Some(tts) = self.tts(&prefs) else {
            let _ = app.emit("luma://caption", &text);
            return;
        };
        match tts.synthesize(&text).await {
            Ok(audio) if self.current(epoch) => {
                if let Some(s) = &self.speaker {
                    let app2 = app.clone();
                    let caption = text.clone();
                    s.send(SpeakerCmd::Callback(Box::new(move || {
                        let _ = app2.emit("luma://caption", &caption);
                    })));
                    s.send(SpeakerCmd::Play(audio));
                }
            }
            Ok(_) => {}
            Err(e) => {
                log::warn!("tts failed: {e}");
                let _ = app.emit("luma://caption", &text);
            }
        }
    }

    fn emit_marker(&self, app: &AppHandle, m: &Marker) {
        match m {
            Marker::Caption(t) => {
                let _ = app.emit("luma://caption", t);
            }
            Marker::Magnify { display, rect, src } => {
                let _ = app.emit_to(
                    format!("overlay-{display}"),
                    "luma://magnify",
                    serde_json::json!({ "display": display, "rect": rect, "src": src }),
                );
            }
            Marker::Annotate(a) => {
                // A refined version may already be waiting for this mark.
                let better = {
                    let mut book = self.refined.lock().unwrap();
                    if book.epoch == self.epoch.load(Ordering::SeqCst) {
                        if let Some(id) = a.id() {
                            book.shown.insert(id.to_string());
                            book.better.get(id).cloned()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                let a = better.as_ref().unwrap_or(a);
                let target = a.display();
                match target {
                    Some(d) => {
                        let _ = app.emit_to(format!("overlay-{d}"), "luma://annotate", a);
                        // (re)arm the watcher; a new mark resets the baseline
                        let key = screen::active_window().map(|w| w.key()).unwrap_or_default();
                        *self.watch.lock().unwrap() =
                            Some(Watch { display: d, window_key: key, baseline: None, last_mark: Instant::now() });
                    }
                    None => {
                        let _ = app.emit("luma://annotate", a);
                    }
                }
            }
        }
    }
}

/// Snaps model boxes to accessibility element frames (exact edges for
/// native controls; unchanged when nothing matches well).
pub fn element_snapper(displays: Vec<luma_core::geometry::Display>, raw: Arc<RawSnapshot>) -> luma_core::annotation::Snapper<'static> {
    Box::new(move |at, label, kind| {
        let d = displays.iter().find(|d| d.index == at.display_index)?;
        let t = Instant::now();
        if kind == luma_core::snap::SnapKind::Text {
            // Text spans snap to on-device OCR lines, on the captured frame.
            if raw.display.index != d.index {
                return None;
            }
            let lines = crate::ocr::lines_near(&raw.full, d, &at.rect);
            let snapped = luma_core::snap::snap_text(&at.rect, &lines);
            log::debug!("text snap: {} lines, {} in {} ms", lines.len(), if snapped.is_some() { "snapped" } else { "kept" }, t.elapsed().as_millis());
            return snapped;
        }
        let elements = crate::ax::elements_at(d.view_to_input(at.rect.center()));
        let snapped = luma_core::snap::snap(&at.rect, label, kind, &elements, d);
        log::debug!(
            "snap {:?} {label:?}: {} candidates, {} in {} ms",
            kind,
            elements.len(),
            if snapped.is_some() { "snapped" } else { "kept" },
            t.elapsed().as_millis()
        );
        snapped
    })
}
