# LUMA

A visual AI companion for macOS and Windows. Hold a shortcut, point at anything
and ask, for example "what is this?". LUMA answers out loud while drawing
boxes, arrows, labels and numbered steps directly over your screen, timed to
what it is saying.

- **Sees** the display you are working on, plus a sharp close-up around your pointer
- **Talks** naturally, and you can interrupt it any time by pressing the shortcut again
- **Points** with boxes, circles, highlights, arrows, step badges, spotlight and a flying pointer
- **Draws its own diagrams:** "sketch how DNS works" puts a whiteboard on screen and builds
  the diagram (boxes, arrows, freehand lines) step by step as it explains
- **Teaches:** "teach me how to add a table" points at one step at a time, watches you do it,
  and corrects you kindly; ask "why?" mid-way and it carries on afterwards
- **Does things** you ask for, asking before anything consequential; "what did you change?"
  and "undo that" work afterwards
- **Snaps** boxes to the exact edges of native controls (macOS Accessibility, Windows UI
  Automation), text highlights to real lines (on-device OCR), and re-checks tiny targets on a close-up
- **Knows what you selected** and which field has focus, and blacks out password fields
  before any screenshot leaves your computer
- **Fast mode** (optional): Gemini Live native audio answers in well under a second
- **Repeats** the last answer, drawings included, when you say "say that again", and
  "never mind" stops, both instantly, without a model call
- **Remembers** the conversation ("explain the one you just showed me", "go deeper")
- **Private by design:** captures only while you hold the shortcut, keys live in your
  OS keychain, password managers are excluded, and nothing is saved to disk

Built with Tauri 2 (Rust + TypeScript), Gemini, AssemblyAI and Sarvam.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/ROADMAP.md](docs/ROADMAP.md)
and [docs/RESEARCH.md](docs/RESEARCH.md).

## Use it

1. Install LUMA (`.dmg` on macOS, `.msi`/`.exe` on Windows) from Releases, or build it yourself (below).
2. Add your keys, either in LUMA → **Settings** (saved to the OS keychain) or in a `.env` file
   (copy `.env.example`; for the installed app put it in `~/Library/Application Support/com.luma.companion/`
   on macOS or `%APPDATA%\com.luma.companion\` on Windows). `.env` takes priority.
   - **Gemini** (required unless you switch models): https://aistudio.google.com/apikey. Use a billing-enabled
     project so your screen content is not used for training.
   - Or switch the answering model in **Settings → Brain → Model provider**:
     **Groq** (needs a key from https://console.groq.com/keys and a model that reads images) or
     **Ollama** (local, no key; pull a vision model such as `ollama pull qwen2.5vl` and keep `ollama serve`
     running). Fast mode (Gemini Live) is Gemini-only; with Groq or Ollama every turn uses the standard pipeline.
   - **AssemblyAI** (speech-to-text; required on Windows. On a Mac, LUMA falls back to
     on-device recognition without it)
   - **Sarvam** (optional, voice). Without it, answers appear as captions.
3. Hold **⌘⇧Space** (macOS) or **Ctrl+Alt+Space** (Windows), ask, release.

macOS asks once for Microphone and Screen Recording permission.

## Develop

Requirements: Node 20+, Rust (stable), and on macOS the Xcode Command Line Tools
(`xcode-select --install`). The full Xcode app is not needed.

```bash
npm install
cp .env.example .env     # then fill in your keys (git-ignored)
npm run tauri dev        # run the app
npm test                 # TypeScript + Rust unit tests
npm run demo             # overlay demo page: http://localhost:1420/dev/demo.html
```

### Eval

```bash
npm run eval:render                              # render fixtures with headless Chrome
npm run eval                                     # score grounding, flows, ambiguity, latency
npm run eval -- --filter editor --repeat 3
npm run eval -- --live gemini-3.8-live          # the same cases through fast mode
LUMA_REPLAY=1 npm run eval -- --filter sketch    # then /dev/replay.html?case=sketch-web-request
```

### Release builds

Push a tag such as `v0.1.0`. GitHub Actions builds macOS (Apple Silicon and
Intel) and Windows installers into a draft release
(`.github/workflows/release.yml`). Unsigned builds work, but show a first-launch
warning. Signing, notarization, auto-update and the optional key proxy are set
up through repository secrets: see [docs/DISTRIBUTION.md](docs/DISTRIBUTION.md).
