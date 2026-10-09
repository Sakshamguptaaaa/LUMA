//! OpenAI-compatible `chat/completions` client (SSE). One client covers Groq
//! (cloud, needs a key) and Ollama (local, no key). Neither goes through the
//! LUMA key proxy.

use anyhow::{anyhow, Result};
use base64::Engine;
use futures_util::StreamExt;
use luma_core::sse::SseParser;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::gemini::{HistoryTurn, ImagePart};

pub const GROQ_URL: &str = "https://api.groq.com/openai/v1";
pub const OLLAMA_URL: &str = "http://localhost:11434";

#[derive(Clone)]
pub struct OpenAiCompat {
    pub client: reqwest::Client,
    /// API root ending in `/v1`, e.g. "https://api.groq.com/openai/v1".
    pub base_url: String,
    /// Empty for servers that need none (Ollama).
    pub api_key: String,
    pub model: String,
    /// Shown in errors: "Groq", "Ollama".
    pub name: &'static str,
}

impl OpenAiCompat {
    pub fn groq(client: reqwest::Client, api_key: String, model: String) -> Self {
        Self { client, base_url: GROQ_URL.into(), api_key, model, name: "Groq" }
    }

    /// `url` is the Ollama server ("http://localhost:11434"), with or without `/v1`.
    pub fn ollama(client: reqwest::Client, url: &str, model: String) -> Self {
        let url = url.trim().trim_end_matches('/');
        let url = if url.is_empty() { OLLAMA_URL } else { url };
        let base_url = if url.ends_with("/v1") { url.to_string() } else { format!("{url}/v1") };
        Self { client, base_url, api_key: String::new(), model, name: "Ollama" }
    }

    pub fn build_body(&self, system: &str, history: &[HistoryTurn], context: &str, images: &[ImagePart], user: &str) -> Value {
        let mut messages = vec![json!({"role": "system", "content": system})];
        for t in history {
            messages.push(json!({"role": "user", "content": t.user}));
            messages.push(json!({"role": "assistant", "content": t.assistant}));
        }
        let mut parts = Vec::new();
        if !context.is_empty() {
            parts.push(json!({"type": "text", "text": context}));
        }
        for img in images {
            let b64 = base64::engine::general_purpose::STANDARD.encode(img.jpeg);
            parts.push(json!({"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}}));
        }
        parts.push(json!({"type": "text", "text": format!("The user says: \"{user}\"")}));
        messages.push(json!({"role": "user", "content": parts}));
        json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "max_tokens": 4096,
        })
    }

    /// Streams visible answer text into `out`. Returns when the response ends.
    pub async fn stream(&self, body: &Value, out: mpsc::UnboundedSender<String>) -> Result<()> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self.client.post(&url).json(body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await.map_err(|e| anyhow!(unreachable(self.name, &self.base_url, &e.to_string())))?;
        let status = resp.status();
        if !status.is_success() {
            let t = resp.text().await.unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&t)
                .ok()
                .and_then(|v| error_message(&v))
                .unwrap_or_else(|| t.chars().take(200).collect());
            return Err(anyhow!(friendly_error(self.name, status.as_u16(), &msg, &self.model)));
        }
        let mut sse = SseParser::new();
        let mut think = ThinkFilter::default();
        let mut stream = resp.bytes_stream();
        let name = self.name;
        let mut handle = |payload: String| -> Result<bool> {
            match parse_chunk(&payload) {
                Chunk::Done => Ok(false),
                Chunk::Error(e) => Err(anyhow!("{name}: {e}")),
                Chunk::Text(t) => {
                    let t = think.push(&t);
                    if !t.is_empty() {
                        let _ = out.send(t);
                    }
                    Ok(true)
                }
            }
        };
        'read: while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow!("{name} stream interrupted: {e}"))?;
            for payload in sse.push(&chunk) {
                if !handle(payload)? {
                    break 'read;
                }
            }
        }
        if let Some(p) = sse.finish() {
            handle(p)?;
        }
        let rest = think.finish();
        if !rest.is_empty() {
            let _ = out.send(rest);
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
enum Chunk {
    Text(String),
    Done,
    Error(String),
}

fn error_message(v: &Value) -> Option<String> {
    let e = v.get("error")?;
    e.get("message").and_then(Value::as_str).or_else(|| e.as_str()).map(str::to_string)
}

fn parse_chunk(payload: &str) -> Chunk {
    if payload.trim() == "[DONE]" {
        return Chunk::Done;
    }
    let Ok(v) = serde_json::from_str::<Value>(payload) else {
        return Chunk::Error("unparseable response chunk".into());
    };
    if let Some(e) = error_message(&v) {
        return Chunk::Error(e);
    }
    let text = v
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Chunk::Text(text.to_string())
}

/// Drops `<think>…</think>` reasoning that some models (Qwen, DeepSeek)
/// stream inline with the answer, even when a tag is split across chunks.
#[derive(Default)]
struct ThinkFilter {
    inside: bool,
    pending: String,
}

impl ThinkFilter {
    fn push(&mut self, s: &str) -> String {
        self.pending.push_str(s);
        let mut out = String::new();
        loop {
            let tag = if self.inside { "</think>" } else { "<think>" };
            if let Some(i) = self.pending.find(tag) {
                if !self.inside {
                    out.push_str(&self.pending[..i]);
                }
                self.pending.drain(..i + tag.len());
                self.inside = !self.inside;
                continue;
            }
            // hold back a possible partial tag at the end
            let keep = (1..tag.len()).rev().find(|&k| self.pending.ends_with(&tag[..k])).unwrap_or(0);
            let cut = self.pending.len() - keep;
            if !self.inside {
                out.push_str(&self.pending[..cut]);
            }
            self.pending.drain(..cut);
            return out;
        }
    }

    fn finish(&mut self) -> String {
        let rest = std::mem::take(&mut self.pending);
        if self.inside {
            String::new()
        } else {
            rest
        }
    }
}

fn unreachable(name: &str, base_url: &str, e: &str) -> String {
    if name == "Ollama" {
        let server = base_url.trim_end_matches("/v1");
        format!("Couldn't reach Ollama at {server}. Start it (ollama serve) or fix the Ollama URL in Settings.")
    } else {
        format!("couldn't reach {name}: {e}")
    }
}

/// Turn API errors into something a user can act on from the status pill.
pub fn friendly_error(name: &str, status: u16, msg: &str, model: &str) -> String {
    let m = msg.to_lowercase();
    if m.contains("image") && (m.contains("support") || m.contains("vision") || m.contains("multimodal")) {
        return format!("{name} model \"{model}\" can't read images. Pick a vision model in Settings.");
    }
    match status {
        401 | 403 => format!("{name} rejected the API key. Check LUMA_GROQ_API_KEY in .env or Settings."),
        404 if name == "Ollama" => format!("Ollama doesn't have \"{model}\". Run: ollama pull {model}"),
        404 => format!("{name} model \"{model}\" wasn't found. Pick another model in Settings."),
        413 => format!("The screenshot is too large for {name}. Lower the image size in Settings."),
        429 => format!("{name} is rate-limiting requests. Wait a moment and try again."),
        s if s >= 500 => format!("{name} is having trouble right now. Try again in a moment."),
        s => format!("{name} error {s}: {}", msg.chars().take(160).collect::<String>()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> OpenAiCompat {
        OpenAiCompat::groq(reqwest::Client::new(), "k".into(), "llama".into())
    }

    #[test]
    fn body_is_openai_chat_with_images() {
        let h = [HistoryTurn { user: "hi", assistant: "hello" }];
        let b = client().build_body("sys", &h, "ctx", &[ImagePart { jpeg: b"jpg" }], "where?");
        assert_eq!(b["model"], "llama");
        assert_eq!(b["stream"], true);
        let m = b["messages"].as_array().unwrap();
        assert_eq!(m.len(), 4);
        assert_eq!(m[0]["role"], "system");
        assert_eq!(m[2]["role"], "assistant");
        let parts = m[3]["content"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "ctx");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/jpeg;base64,anBn");
        assert_eq!(parts[2]["text"], "The user says: \"where?\"");
    }

    #[test]
    fn ollama_url_gets_v1() {
        let c = reqwest::Client::new();
        assert_eq!(OpenAiCompat::ollama(c.clone(), "http://localhost:11434/", "m".into()).base_url, "http://localhost:11434/v1");
        assert_eq!(OpenAiCompat::ollama(c.clone(), "http://box:11434/v1", "m".into()).base_url, "http://box:11434/v1");
        assert_eq!(OpenAiCompat::ollama(c, " ", "m".into()).base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn parses_stream_chunks() {
        assert_eq!(parse_chunk(r#"{"choices":[{"delta":{"content":"Hi"}}]}"#), Chunk::Text("Hi".into()));
        assert_eq!(parse_chunk(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#), Chunk::Text(String::new()));
        assert_eq!(parse_chunk("[DONE]"), Chunk::Done);
        assert_eq!(parse_chunk(r#"{"error":{"message":"boom"}}"#), Chunk::Error("boom".into()));
    }

    #[test]
    fn strips_think_blocks_across_chunks() {
        let mut f = ThinkFilter::default();
        let mut out = String::new();
        for c in ["<thi", "nk>plan", "ning</th", "ink>Click ", "the <b>", "Save</b> button<"] {
            out.push_str(&f.push(c));
        }
        out.push_str(&f.finish());
        assert_eq!(out, "Click the <b>Save</b> button<");
    }

    #[test]
    fn errors_are_actionable() {
        assert!(friendly_error("Groq", 401, "Invalid API Key", "m").contains("API key"));
        assert!(friendly_error("Ollama", 404, "model not found", "llava").contains("ollama pull llava"));
        assert!(friendly_error("Groq", 400, "this model does not support image input", "m").contains("vision model"));
        assert!(friendly_error("Groq", 429, "slow down", "m").contains("rate-limiting"));
    }
}
