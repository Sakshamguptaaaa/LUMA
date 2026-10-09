import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

type Provider = "gemini" | "assemblyai" | "sarvam" | "proxy";

interface Prefs {
  hotkey: string;
  gemini_model: string;
  thinking_level: string;
  stt_model: string;
  tts_speaker: string;
  tts_language: string;
  tts_pace: number;
  max_image_edge: number;
  send_closeup: boolean;
  excluded_apps: string[];
  paused: boolean;
  annotate: string;
  long_press_ms: number;
  gesture: boolean;
  snap_to_elements: boolean;
  refine_small: boolean;
  proxy_url: string;
  voice_engine: string;
  live_model: string;
  follow_up: boolean;
  warm_stt: boolean;
  remember_lessons: boolean;
  autopilot: boolean;
  max_task_steps: number;
}
interface SettingsView {
  prefs: Prefs;
  keys: Record<Provider, "env" | "keychain" | "proxy" | null>;
  platform: string;
}

const PROVIDERS: { id: Provider; name: string; help: string; required: boolean }[] = [
  { id: "gemini", name: "Gemini", help: "aistudio.google.com → Get API key", required: true },
  { id: "assemblyai", name: "AssemblyAI", help: "Speech-to-text. assemblyai.com/dashboard", required: true },
  { id: "sarvam", name: "Sarvam", help: "Voice. dashboard.sarvam.ai (optional: captions only without it)", required: false },
  { id: "proxy", name: "LUMA proxy token", help: "Only if your team runs a LUMA key proxy (set its URL below)", required: false },
];

const SPEAKERS =
  "shubh aditya ritu priya neha rahul pooja rohan simran kavya amit dev ishita shreya ratan varun manan sumit roopa kabir aayan ashutosh advait anand tanya tarun sunny mani gokul vijay shruti suhani mohit kavitha rehan soham rupali".split(
    " ",
  );
const LANGUAGES: [string, string][] = [
  ["en-IN", "English (India)"],
  ["hi-IN", "Hindi"],
  ["bn-IN", "Bengali"],
  ["ta-IN", "Tamil"],
  ["te-IN", "Telugu"],
  ["kn-IN", "Kannada"],
  ["ml-IN", "Malayalam"],
  ["mr-IN", "Marathi"],
  ["gu-IN", "Gujarati"],
  ["pa-IN", "Punjabi"],
  ["od-IN", "Odia"],
];

const MODES: Record<string, { model: string; thinking: string }> = {
  accurate: { model: "gemini-3.8-flash", thinking: "low" },
  fast: { model: "gemini-3.5-flash", thinking: "minimal" },
};
const modeOf = (model: string, thinking: string) =>
  Object.entries(MODES).find(([, m]) => m.model === model && m.thinking === thinking)?.[0] ?? "custom";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
let view: SettingsView;

function addMsg(kind: "you" | "luma" | "sys", text: string) {
  const log = $("log");
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

function prettyHotkey(h: string, platform: string) {
  const mac = platform === "macos";
  return h
    .split("+")
    .map((k) => {
      const m: Record<string, string> = mac
        ? { Command: "⌘", CommandOrControl: "⌘", Shift: "⇧", Alt: "⌥", Option: "⌥", Control: "⌃", Space: "Space" }
        : { CommandOrControl: "Ctrl", Control: "Ctrl" };
      return `<kbd>${m[k] ?? k}</kbd>`;
    })
    .join(" ");
}

function render() {
  const p = view.prefs;
  // On a Mac, speech can fall back to on-device recognition.
  const missing = PROVIDERS.filter((x) => x.required && !view.keys[x.id] && !(x.id === "assemblyai" && view.platform === "macos"));
  $("hint").innerHTML = missing.length
    ? `To start, add your ${missing.map((m) => m.name).join(" and ")} key in <b>Settings</b>.`
    : `Hold ${prettyHotkey(p.hotkey, view.platform)} and ask, then release. Say “show me” and LUMA draws on your screen.${p.gesture ? ` Or hold the trackpad still on something for ${(p.long_press_ms / 1000).toFixed(1)} s and ask about it.` : ""}${p.autopilot ? " Ask it to <i>do</i> something (“change my GitHub username”) and it will, asking before anything important. Say “teach me how to…” and it guides you one step at a time while you do it." : " Ask it to do something (“change my GitHub username”) and it guides you one step at a time while you do it."} Press again to stop.`;

  const keys = $("keys");
  keys.innerHTML = "";
  for (const prov of PROVIDERS) {
    const row = document.createElement("div");
    row.className = "keyrow";
    const src = view.keys[prov.id];
    const ok = src !== null;
    const status =
      src === "env" ? "✓ from .env" : src === "keychain" ? "✓ saved" : src === "proxy" ? "✓ via proxy" : prov.required ? "required" : "optional";
    row.innerHTML = `<label>${prov.name} <span class="${ok ? "ok" : "missing"}">${status}</span>
        <input type="password" autocomplete="off" spellcheck="false" placeholder="${src === "env" ? "Set in .env (takes priority)" : ok ? "Paste to replace" : prov.help}" /></label>
        <button class="ghost">Save</button>`;
    const input = row.querySelector("input")!;
    row.querySelector("button")!.addEventListener("click", async () => {
      await invoke("save_key", { provider: prov.id, key: input.value });
      input.value = "";
      await load();
    });
    keys.appendChild(row);
  }

  const sp = $<HTMLSelectElement>("tts_speaker");
  sp.innerHTML = SPEAKERS.map((s) => `<option value="${s}">${s[0].toUpperCase() + s.slice(1)}</option>`).join("");
  sp.value = p.tts_speaker;
  const lang = $<HTMLSelectElement>("tts_language");
  lang.innerHTML = LANGUAGES.map(([v, n]) => `<option value="${v}">${n}</option>`).join("");
  lang.value = p.tts_language;
  $<HTMLInputElement>("tts_pace").value = String(p.tts_pace);
  $<HTMLInputElement>("gemini_model").value = p.gemini_model;
  $<HTMLSelectElement>("thinking_level").value = p.thinking_level;
  $<HTMLSelectElement>("mode").value = modeOf(p.gemini_model, p.thinking_level);
  $<HTMLInputElement>("stt_model").value = p.stt_model;
  $<HTMLInputElement>("hotkey").value = p.hotkey;
  $<HTMLInputElement>("send_closeup").checked = p.send_closeup;
  $<HTMLSelectElement>("annotate").value = p.annotate;
  $<HTMLInputElement>("long_press_s").value = (p.long_press_ms / 1000).toFixed(1);
  $<HTMLInputElement>("gesture").checked = p.gesture;
  $<HTMLInputElement>("snap_to_elements").checked = p.snap_to_elements;
  $<HTMLInputElement>("refine_small").checked = p.refine_small;
  $<HTMLInputElement>("proxy_url").value = p.proxy_url;
  $<HTMLSelectElement>("voice_engine").value = p.voice_engine;
  $<HTMLInputElement>("follow_up").checked = p.follow_up;
  $<HTMLInputElement>("warm_stt").checked = p.warm_stt;
  $<HTMLInputElement>("remember_lessons").checked = p.remember_lessons;
  $<HTMLInputElement>("autopilot").checked = p.autopilot;
  $<HTMLInputElement>("paused").checked = p.paused;
  $<HTMLTextAreaElement>("excluded_apps").value = p.excluded_apps.join("\n");

  $("perm").innerHTML =
    view.platform === "macos"
      ? `macOS asks once for <b>Microphone</b> and <b>Screen Recording</b>. If answers say the screen is blank, enable LUMA in System Settings → Privacy &amp; Security → Screen &amp; System Audio Recording, then reopen LUMA.`
      : `Windows needs microphone access (Settings → Privacy → Microphone). LUMA cannot see or point into apps running as administrator.`;
}

function collect(): Prefs {
  return {
    ...view.prefs,
    tts_speaker: $<HTMLSelectElement>("tts_speaker").value,
    tts_language: $<HTMLSelectElement>("tts_language").value,
    tts_pace: Number($<HTMLInputElement>("tts_pace").value),
    gemini_model: $<HTMLInputElement>("gemini_model").value.trim(),
    thinking_level: $<HTMLSelectElement>("thinking_level").value,
    stt_model: $<HTMLInputElement>("stt_model").value.trim(),
    hotkey: $<HTMLInputElement>("hotkey").value.trim(),
    send_closeup: $<HTMLInputElement>("send_closeup").checked,
    annotate: $<HTMLSelectElement>("annotate").value,
    long_press_ms: Math.round(Math.min(5, Math.max(0.6, Number($<HTMLInputElement>("long_press_s").value) || 2)) * 1000),
    gesture: $<HTMLInputElement>("gesture").checked,
    snap_to_elements: $<HTMLInputElement>("snap_to_elements").checked,
    refine_small: $<HTMLInputElement>("refine_small").checked,
    proxy_url: $<HTMLInputElement>("proxy_url").value.trim(),
    voice_engine: $<HTMLSelectElement>("voice_engine").value,
    follow_up: $<HTMLInputElement>("follow_up").checked,
    warm_stt: $<HTMLInputElement>("warm_stt").checked,
    remember_lessons: $<HTMLInputElement>("remember_lessons").checked,
    autopilot: $<HTMLInputElement>("autopilot").checked,
    excluded_apps: $<HTMLTextAreaElement>("excluded_apps")
      .value.split("\n")
      .map((s) => s.trim())
      .filter(Boolean),
  };
}

async function save(note: string) {
  try {
    await invoke("save_prefs", { prefs: collect() });
    $(note).textContent = "Saved";
    await load();
  } catch (e) {
    $(note).textContent = String(e);
  }
  setTimeout(() => ($(note).textContent = ""), 2500);
}

async function load() {
  view = await invoke<SettingsView>("get_settings");
  render();
}

// tabs
document.querySelectorAll<HTMLButtonElement>(".tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll<HTMLButtonElement>(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    document.querySelectorAll<HTMLElement>("[data-panel]").forEach((s) => (s.hidden = s.dataset.panel !== b.dataset.tab));
  }),
);

$("mode").addEventListener("change", (e) => {
  const m = MODES[(e.target as HTMLSelectElement).value];
  if (!m) return;
  $<HTMLInputElement>("gemini_model").value = m.model;
  $<HTMLSelectElement>("thinking_level").value = m.thinking;
});
const syncMode = () =>
  ($<HTMLSelectElement>("mode").value = modeOf(
    $<HTMLInputElement>("gemini_model").value.trim(),
    $<HTMLSelectElement>("thinking_level").value,
  ));
$("gemini_model").addEventListener("input", syncMode);
$("thinking_level").addEventListener("change", syncMode);

$("save").addEventListener("click", () => save("saved"));
$("save2").addEventListener("click", () => save("saved2"));
$("clear").addEventListener("click", async () => {
  await invoke("clear_session");
  $("log").innerHTML = "";
  $("saved2").textContent = "Forgotten";
});
$<HTMLInputElement>("paused").addEventListener("change", (e) =>
  invoke("set_paused", { paused: (e.target as HTMLInputElement).checked }),
);
$("stop").addEventListener("click", () => invoke("stop"));
$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $<HTMLInputElement>("question");
  if (!q.value.trim()) return;
  invoke("ask", { question: q.value });
  q.value = "";
});

const STATUS: Record<string, string> = {
  idle: "Ready",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Talking",
  acting: "Working on a task",
  waiting: "Waiting for your answer",
  paused: "Paused",
  error: "Problem",
};
listen<{ phase: string; message?: string | null }>("luma://status", (e) => {
  $("status").textContent = STATUS[e.payload.phase] ?? e.payload.phase;
  if (e.payload.message) addMsg("sys", e.payload.message);
});
listen<string>("luma://question", (e) => addMsg("you", e.payload));
listen<string>("luma://answer", (e) => e.payload && addMsg("luma", e.payload));
listen<string>("luma://notice", (e) => addMsg("sys", e.payload));
listen<{ step: number; say: string; action: string }>("luma://step", (e) =>
  addMsg("sys", `Step ${e.payload.step}: ${e.payload.action}${e.payload.say ? ` (“${e.payload.say}”)` : ""}`),
);

load();
