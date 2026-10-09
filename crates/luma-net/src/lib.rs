//! Provider clients. Keys are passed in by the caller and never logged.

pub mod assemblyai;
pub mod gemini;
pub mod sarvam;
pub mod sarvam_ws;
pub mod live;
pub mod llm;
pub mod openai;
pub mod route;
pub mod vision;

/// Audio sample rate expected by the streaming speech-to-text.
pub const STT_RATE: u32 = 16_000;
