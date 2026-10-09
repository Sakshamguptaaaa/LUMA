//! Gemini `streamGenerateContent` client (SSE).

use anyhow::{anyhow, Result};
use base64::Engine;
use futures_util::StreamExt;
use luma_core::sse::{parse_gemini_chunk, SseParser};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::llm::Llm;

#[derive(Clone)]
pub struct Gemini {
    pub client: reqwest::Client,
    pub api_key: String,
    pub model: String,
    pub thinking_level: String,
    /// "MEDIA_RESOLUTION_HIGH" for precise pointing, MEDIUM/LOW to save tokens.
    pub media_resolution: &'static str,
}

pub struct ImagePart<'a> {
    pub jpeg: &'a [u8],
}

pub struct HistoryTurn<'a> {
    pub user: &'a str,
    pub assistant: &'a str,
}

impl Gemini {
    pub fn build_body(
        &self,
        system: &str,
        history: &[HistoryTurn],
        context: &str,
        images: &[ImagePart],
        user: &str,
    ) -> Value {
        let mut contents: Vec<Value> = Vec::new();
        for t in history {
            contents.push(json!({"role": "user", "parts": [{"text": t.user}]}));
            contents.push(json!({"role": "model", "parts": [{"text": t.assistant}]}));
        }
        let mut parts = vec![json!({"text": context})];
        for img in images {
            parts.push(json!({"inlineData": {
                "mimeType": "image/jpeg",
                "data": base64::engine::general_purpose::STANDARD.encode(img.jpeg),
            }}));
        }
        parts.push(json!({"text": format!("The user says: \"{user}\"")}));
        contents.push(json!({"role": "user", "parts": parts}));
        json!({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": self.thinking_level},
                "mediaResolution": self.media_resolution,
                "maxOutputTokens": 4096,
            },
        })
    }

    /// See [`Llm::hedged`]; kept for Gemini-only callers (evals).
    pub fn hedged(
        self,
        body: Value,
        fallback: Option<(Gemini, Value)>,
        after: std::time::Duration,
        out: mpsc::UnboundedSender<String>,
    ) -> tokio::task::JoinHandle<Result<String>> {
        Llm::Gemini(self).hedged(body, fallback.map(|(g, b)| (Llm::Gemini(g), b)), after, out)
    }

    /// Non-streaming convenience: the whole visible answer.
    pub async fn complete(&self, body: &Value) -> Result<String> {
        Llm::Gemini(self.clone()).complete(body).await
    }

    /// Streams visible answer text into `out`. Returns when the response ends.
    pub async fn stream(&self, body: &Value, out: mpsc::UnboundedSender<String>) -> Result<()> {
        let target = crate::route::gemini(&format!("v1beta/models/{}:streamGenerateContent?alt=sse", self.model), &self.api_key);
        let url = target.url;
        let mut body = std::borrow::Cow::Borrowed(body);
        let mut retried = false;
        let resp = loop {
            let resp = self
                .client
                .post(&url)
                .header(target.header.0, &target.header.1)
                .json(body.as_ref())
                .send()
                .await
                .map_err(|e| anyhow!("couldn't reach Gemini: {e}"))?;
            let status = resp.status();
            if status.is_success() {
                break resp;
            }
            let t = resp.text().await.unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&t)
                .ok()
                .and_then(|v| v.pointer("/error/message").and_then(Value::as_str).map(str::to_string))
                .unwrap_or_else(|| t.chars().take(200).collect());
            // Supported thinking levels differ per model; fall back to "low"
            // rather than failing the user's turn.
            if !retried && status.as_u16() == 400 && msg.to_lowercase().contains("thinking level") {
                retried = true;
                let mut b = body.into_owned();
                b["generationConfig"]["thinkingConfig"]["thinkingLevel"] = Value::from("low");
                body = std::borrow::Cow::Owned(b);
                continue;
            }
            return Err(anyhow!(friendly_error(status.as_u16(), &msg, &self.model)));
        };
        let mut sse = SseParser::new();
        let mut stream = resp.bytes_stream();
        let handle = |payload: String| -> Result<()> {
            let d = parse_gemini_chunk(&payload);
            if let Some(e) = d.error {
                return Err(anyhow!("Gemini: {e}"));
            }
            if let Some(b) = d.block_reason {
                return Err(anyhow!("Gemini declined to answer ({b})"));
            }
            if !d.text.is_empty() {
                let _ = out.send(d.text);
            }
            Ok(())
        };
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow!("Gemini stream interrupted: {e}"))?;
            for payload in sse.push(&chunk) {
                handle(payload)?;
            }
        }
        if let Some(p) = sse.finish() {
            handle(p)?;
        }
        Ok(())
    }
}

/// Turn API errors into something a user can act on from the status pill.
pub fn friendly_error(status: u16, msg: &str, model: &str) -> String {
    let m = msg.to_lowercase();
    match status {
        429 if m.contains("spending cap") || m.contains("spend cap") => {
            "Gemini's monthly spending cap is reached. Raise it at ai.studio/spend, then try again.".into()
        }
        429 => "Gemini is rate-limiting requests. Wait a moment and try again.".into(),
        400 if m.contains("api key") => "Gemini rejected the API key. Check LUMA_GEMINI_API_KEY in .env or Settings.".into(),
        401 | 403 => "Gemini rejected the API key. Check LUMA_GEMINI_API_KEY in .env or Settings.".into(),
        404 => format!("Gemini model \"{model}\" wasn't found. Pick another model in Settings."),
        s if s >= 500 => "Gemini is having trouble right now. Try again in a moment.".into(),
        s => format!("Gemini error {s}: {}", msg.chars().take(160).collect::<String>()),
    }
}

#[cfg(test)]
mod tests {
    use super::friendly_error;

    #[test]
    fn errors_are_actionable() {
        assert!(friendly_error(429, "Your project has exceeded its monthly spending cap.", "m").contains("ai.studio/spend"));
        assert!(friendly_error(429, "Resource exhausted", "m").contains("rate-limiting"));
        assert!(friendly_error(403, "denied", "m").contains("API key"));
        assert!(friendly_error(404, "not found", "gemini-x").contains("gemini-x"));
        assert!(friendly_error(503, "overloaded", "m").contains("trouble"));
    }
}
