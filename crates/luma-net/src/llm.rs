//! One interface over the answering models: Gemini, or any OpenAI-compatible
//! chat API (Groq, Ollama). Callers build a request body for the model they
//! have, then stream, complete or hedge it without caring which it is.

use anyhow::{anyhow, Result};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::gemini::{Gemini, HistoryTurn, ImagePart};
use crate::openai::OpenAiCompat;

#[derive(Clone)]
pub enum Llm {
    Gemini(Gemini),
    OpenAi(OpenAiCompat),
}

impl From<Gemini> for Llm {
    fn from(g: Gemini) -> Self {
        Llm::Gemini(g)
    }
}

impl From<OpenAiCompat> for Llm {
    fn from(o: OpenAiCompat) -> Self {
        Llm::OpenAi(o)
    }
}

impl Llm {
    pub fn model(&self) -> &str {
        match self {
            Llm::Gemini(g) => &g.model,
            Llm::OpenAi(o) => &o.model,
        }
    }

    /// A request body in this model's own wire format.
    pub fn build_body(&self, system: &str, history: &[HistoryTurn], context: &str, images: &[ImagePart], user: &str) -> Value {
        match self {
            Llm::Gemini(g) => g.build_body(system, history, context, images, user),
            Llm::OpenAi(o) => o.build_body(system, history, context, images, user),
        }
    }

    /// Streams visible answer text into `out`. Returns when the response ends.
    pub async fn stream(&self, body: &Value, out: mpsc::UnboundedSender<String>) -> Result<()> {
        match self {
            Llm::Gemini(g) => g.stream(body, out).await,
            Llm::OpenAi(o) => o.stream(body, out).await,
        }
    }

    /// Non-streaming convenience: the whole visible answer.
    pub async fn complete(&self, body: &Value) -> Result<String> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let this = self.clone();
        let body = body.clone();
        let task = tokio::spawn(async move { this.stream(&body, tx).await });
        let mut out = String::new();
        while let Some(d) = rx.recv().await {
            out.push_str(&d);
        }
        task.await.map_err(|e| anyhow!("{e}"))??;
        Ok(out)
    }

    /// Stream from `self`; if nothing has arrived after `after`, also start
    /// `fallback` (a faster model) and keep whichever produces text first,
    /// cancelling the other. Bounds the worst case when a model deliberates
    /// for a long time. Returns the model that answered.
    pub fn hedged(
        self,
        body: Value,
        fallback: Option<(Llm, Value)>,
        after: std::time::Duration,
        out: mpsc::UnboundedSender<String>,
    ) -> tokio::task::JoinHandle<Result<String>> {
        tokio::spawn(async move {
            let (ptx, mut prx) = mpsc::unbounded_channel::<String>();
            let primary_model = self.model().to_string();
            let p = {
                let g = self.clone();
                tokio::spawn(async move { g.stream(&body, ptx).await })
            };
            // relay one channel to `out` until it ends, then report its result
            async fn relay(
                first: String,
                mut rx: mpsc::UnboundedReceiver<String>,
                task: tokio::task::JoinHandle<Result<()>>,
                out: &mpsc::UnboundedSender<String>,
            ) -> Result<()> {
                let _ = out.send(first);
                while let Some(d) = rx.recv().await {
                    if out.send(d).is_err() {
                        task.abort();
                        return Ok(());
                    }
                }
                task.await.map_err(|e| anyhow!("{e}"))?
            }
            match tokio::time::timeout(after, prx.recv()).await {
                Ok(Some(d)) => return relay(d, prx, p, &out).await.map(|_| primary_model),
                Ok(None) => return p.await.map_err(|e| anyhow!("{e}"))?.map(|_| primary_model),
                Err(_) => {}
            }
            let Some((fb, fbody)) = fallback else {
                return match prx.recv().await {
                    Some(d) => relay(d, prx, p, &out).await.map(|_| primary_model),
                    None => p.await.map_err(|e| anyhow!("{e}"))?.map(|_| primary_model),
                };
            };
            let fb_model = fb.model().to_string();
            let (ftx, mut frx) = mpsc::unbounded_channel::<String>();
            let f = tokio::spawn(async move { fb.stream(&fbody, ftx).await });
            let (mut p_open, mut f_open) = (true, true);
            loop {
                tokio::select! {
                    d = prx.recv(), if p_open => match d {
                        Some(d) => { f.abort(); return relay(d, prx, p, &out).await.map(|_| primary_model); }
                        None => p_open = false,
                    },
                    d = frx.recv(), if f_open => match d {
                        Some(d) => { p.abort(); return relay(d, frx, f, &out).await.map(|_| fb_model); }
                        None => f_open = false,
                    },
                    else => break,
                }
            }
            // neither produced text: surface the primary's error, if any
            p.await.map_err(|e| anyhow!("{e}"))??;
            f.await.map_err(|e| anyhow!("{e}"))??;
            Ok(primary_model)
        })
    }
}
