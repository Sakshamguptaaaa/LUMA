//! API keys live in the OS credential store (macOS Keychain / Windows
//! Credential Manager). Non-secret preferences live in a JSON file in the app
//! config directory. Keys are never logged, never sent to the webview, and
//! never written to disk by LUMA itself.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const SERVICE: &str = "com.luma.companion";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Gemini,
    Groq,
    Assemblyai,
    Sarvam,
    /// Per-device token for a LUMA key proxy (optional).
    Proxy,
}

impl Provider {
    pub const ALL: [Provider; 5] = [Provider::Gemini, Provider::Groq, Provider::Assemblyai, Provider::Sarvam, Provider::Proxy];

    fn account(self) -> &'static str {
        match self {
            Provider::Gemini => "gemini-api-key",
            Provider::Groq => "groq-api-key",
            Provider::Assemblyai => "assemblyai-api-key",
            Provider::Sarvam => "sarvam-api-key",
            Provider::Proxy => "proxy-token",
        }
    }

    /// Environment variables (or `.env` entries) checked before the keychain.
    fn env_vars(self) -> [&'static str; 2] {
        match self {
            Provider::Gemini => ["LUMA_GEMINI_API_KEY", "GEMINI_API_KEY"],
            Provider::Groq => ["LUMA_GROQ_API_KEY", "GROQ_API_KEY"],
            Provider::Assemblyai => ["LUMA_ASSEMBLYAI_API_KEY", "ASSEMBLYAI_API_KEY"],
            Provider::Sarvam => ["LUMA_SARVAM_API_KEY", "SARVAM_API_KEY"],
            Provider::Proxy => ["LUMA_PROXY_TOKEN", "LUMA_PROXY_TOKEN"],
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Provider::Gemini => "Gemini",
            Provider::Groq => "Groq",
            Provider::Assemblyai => "AssemblyAI",
            Provider::Sarvam => "Sarvam",
            Provider::Proxy => "LUMA proxy",
        }
    }

    /// Whether a configured LUMA key proxy holds this provider's key. The
    /// proxy only serves Gemini, AssemblyAI and Sarvam.
    fn via_proxy(self) -> bool {
        matches!(self, Provider::Gemini | Provider::Assemblyai | Provider::Sarvam) && luma_net::route::proxy().is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum KeySource {
    Env,
    Keychain,
    /// Provided by the configured LUMA key proxy.
    Proxy,
}

/// Load `.env` files without overriding variables already set in the real
/// environment. First file wins for any given key:
/// 1. `.env` in the working directory
/// 2. the repository's `.env` (debug builds only, so `npm run tauri dev` works)
/// 3. `.env` in LUMA's config directory (for installed builds)
pub fn load_env_files(config_dir: Option<&std::path::Path>) {
    let _ = dotenvy::dotenv();
    #[cfg(debug_assertions)]
    let _ = dotenvy::from_path(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.env"));
    if let Some(dir) = config_dir {
        let _ = dotenvy::from_path(dir.join(".env"));
    }
}

fn env_key(p: Provider) -> Option<String> {
    p.env_vars()
        .iter()
        .filter_map(|v| std::env::var(v).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

pub fn key_source(p: Provider) -> Option<KeySource> {
    if p.via_proxy() {
        return Some(KeySource::Proxy);
    }
    if env_key(p).is_some() {
        Some(KeySource::Env)
    } else if keychain_key(p).is_some() {
        Some(KeySource::Keychain)
    } else {
        None
    }
}

/// `.env` / environment first, then the OS keychain. With a key proxy
/// configured, provider keys live on the proxy: a placeholder is returned
/// and the network clients authenticate with the device token instead.
pub fn get_key(p: Provider) -> Option<String> {
    if p.via_proxy() {
        return Some("via-luma-proxy".into());
    }
    env_key(p).or_else(|| keychain_key(p))
}

/// Point the network clients at the proxy in `prefs` (or straight at the
/// providers when none is set).
pub fn apply_proxy(prefs: &Prefs) {
    let token = env_key(Provider::Proxy).or_else(|| keychain_key(Provider::Proxy));
    luma_net::route::set_proxy(token.map(|token| luma_net::route::Proxy { url: prefs.proxy_url.clone(), token }));
}

fn keychain_key(p: Provider) -> Option<String> {
    keyring::Entry::new(SERVICE, p.account())
        .ok()?
        .get_password()
        .ok()
        .filter(|k| !k.trim().is_empty())
}

pub fn set_key(p: Provider, key: &str) -> anyhow::Result<()> {
    let entry = keyring::Entry::new(SERVICE, p.account())?;
    let key = key.trim();
    if key.is_empty() {
        let _ = entry.delete_credential();
    } else {
        entry.set_password(key)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// Hold to talk. Uses the global-shortcut plugin's accelerator syntax.
    pub hotkey: String,
    /// Which model answers: "gemini", "groq" or "ollama". Gemini-only
    /// features (fast mode with Gemini Live, the fast-model fallback) are
    /// off with the others.
    pub llm_provider: String,
    pub gemini_model: String,
    /// minimal | low | medium | high
    pub thinking_level: String,
    pub stt_model: String,
    pub tts_speaker: String,
    pub tts_language: String,
    pub tts_pace: f32,
    /// Long edge of the full-screen image sent to the model, in pixels.
    pub max_image_edge: u32,
    /// Also send a high-detail close-up around the pointer.
    pub send_closeup: bool,
    /// LUMA will not look at the screen while one of these apps is in front.
    pub excluded_apps: Vec<String>,
    /// When paused, the hotkey does nothing and nothing is captured.
    pub paused: bool,
    /// Drawing on screen. "auto": shortcut turns draw only when asked or
    /// clearly useful, long-press turns always draw. "always" / "never".
    pub annotate: String,
    /// Hold the trackpad/mouse still this long to point-and-ask.
    pub long_press_ms: u64,
    /// Point-and-ask by long-pressing the trackpad/mouse.
    pub gesture: bool,
    /// Model for quick shortcut turns (speed first).
    pub fast_model: String,
    pub fast_thinking: String,
    /// Snap boxes to the exact frames of native controls (Accessibility).
    pub snap_to_elements: bool,
    /// Re-check small marks (icons, thin rows) on a close-up crop.
    pub refine_small: bool,
    /// "standard" (speech-to-text → Gemini → voice, every feature) or "live"
    /// (fast mode: Gemini Live native audio for questions; tasks and lessons
    /// still use the standard pipeline).
    pub voice_engine: String,
    pub live_model: String,
    /// Keep a speech-to-text session open for a few seconds after each
    /// answer, so a follow-up question starts streaming instantly.
    pub warm_stt: bool,
    /// After an answer, listen hands-free for a few seconds for a follow-up
    /// (the mic opens only once LUMA has stopped talking).
    pub follow_up: bool,
    /// Save lesson progress (text only) so "continue from where we left
    /// off" works after quitting LUMA.
    pub remember_lessons: bool,
    /// Let LUMA click and type to carry out tasks you ask for.
    pub can_act: bool,
    /// Upper bound on agent steps per task.
    pub max_task_steps: usize,
    /// Optional LUMA key proxy (keys held by your team, not on this computer).
    pub proxy_url: String,
    /// Groq model (must accept images), used for every turn.
    pub groq_model: String,
    /// Ollama model (must accept images), used for every turn.
    pub ollama_model: String,
    /// Ollama server, e.g. "http://localhost:11434".
    pub ollama_url: String,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            hotkey: if cfg!(target_os = "macos") { "Command+Shift+Space" } else { "Control+Alt+Space" }.into(),
            llm_provider: "gemini".into(),
            gemini_model: "gemini-3.8-flash".into(),
            thinking_level: "low".into(),
            stt_model: "universal-3-5-pro".into(),
            tts_speaker: "shubh".into(),
            tts_language: "en-IN".into(),
            tts_pace: 1.1,
            max_image_edge: 1920,
            send_closeup: true,
            excluded_apps: [
                "1Password",
                "Bitwarden",
                "Keychain Access",
                "LastPass",
                "Dashlane",
                "KeePassXC",
                "Passwords",
            ]
            .map(String::from)
            .to_vec(),
            paused: false,
            annotate: "auto".into(),
            long_press_ms: 2000,
            gesture: true,
            fast_model: "gemini-3.5-flash".into(),
            fast_thinking: "minimal".into(),
            snap_to_elements: true,
            refine_small: true,
            remember_lessons: false,
            voice_engine: "standard".into(),
            live_model: "gemini-3.8-live".into(),
            warm_stt: true,
            follow_up: false,
            can_act: true,
            max_task_steps: 25,
            proxy_url: String::new(),
            groq_model: "meta-llama/llama-4-scout-17b-16e-instruct".into(),
            ollama_model: "qwen2.5vl".into(),
            ollama_url: luma_net::openai::OLLAMA_URL.into(),
        }
    }
}

impl Prefs {
    pub fn load(path: &PathBuf) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn uses_gemini(&self) -> bool {
        !matches!(self.llm_provider.as_str(), "groq" | "ollama")
    }

    /// The key the answering model needs, if any (Ollama needs none).
    pub fn llm_key(&self) -> Option<Provider> {
        match self.llm_provider.as_str() {
            "groq" => Some(Provider::Groq),
            "ollama" => None,
            _ => Some(Provider::Gemini),
        }
    }

    pub fn is_excluded(&self, app: &str) -> bool {
        let a = app.to_lowercase();
        self.excluded_apps.iter().any(|e| !e.trim().is_empty() && a.contains(&e.trim().to_lowercase()))
    }
}
