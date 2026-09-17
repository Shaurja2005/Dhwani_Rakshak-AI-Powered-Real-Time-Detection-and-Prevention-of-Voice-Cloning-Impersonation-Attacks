// Gateway client for the UI. The API key lives only in sessionStorage (cleared when the tab closes).

const KEY = "vg.apiKey";
const TENANT = "vg.tenant";

function store() {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function getSettings() {
  const s = store();
  return { apiKey: s?.getItem(KEY) ?? "", tenant: s?.getItem(TENANT) ?? "demo" };
}

export function saveSettings({ apiKey, tenant }) {
  const s = store();
  s?.setItem(KEY, apiKey ?? "");
  s?.setItem(TENANT, tenant ?? "demo");
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
  }
}

export async function api(method, path, body) {
  const { apiKey } = getSettings();
  const res = await fetch(path, {
    method,
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? text);
  return data;
}

export function wsUrl(path) {
  const { apiKey } = getSettings();
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const sep = path.includes("?") ? "&" : "?";
  return `${proto}//${location.host}${path}${sep}api_key=${encodeURIComponent(apiKey)}`;
}

export function uuid() {
  return crypto.randomUUID();
}

export function callMetadata(tenant, extra = {}) {
  return {
    session_id: uuid(),
    tenant_id: tenant,
    direction: "inbound",
    started_at: new Date().toISOString(),
    channel: "webrtc",
    codec_hint: "pcm",
    source_sample_rate: 16000,
    consent_basis: "legitimate_use",
    ...extra,
  };
}

/** Decode any browser-playable audio file to 16 kHz mono Float32 samples. */
export async function decodeTo16k(file) {
  const buf = await file.arrayBuffer();
  const tmp = new AudioContext();
  const decoded = await tmp.decodeAudioData(buf);
  await tmp.close();
  const frames = Math.ceil(decoded.duration * 16000);
  const off = new OfflineAudioContext(1, frames, 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

export function floatTo16BitPCM(samples) {
  const out = new DataView(new ArrayBuffer(samples.length * 2));
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(out.buffer);
}

/** WAV (16-bit mono) bytes → base64, for the file-analysis and enrollment endpoints. */
export function wavBase64(samples16k) {
  const pcm = floatTo16BitPCM(samples16k);
  const header = new DataView(new ArrayBuffer(44));
  const w = (o, s) => [...s].forEach((c, i) => header.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF");
  header.setUint32(4, 36 + pcm.length, true);
  w(8, "WAVE");
  w(12, "fmt ");
  header.setUint32(16, 16, true);
  header.setUint16(20, 1, true);
  header.setUint16(22, 1, true);
  header.setUint32(24, 16000, true);
  header.setUint32(28, 32000, true);
  header.setUint16(32, 2, true);
  header.setUint16(34, 16, true);
  w(36, "data");
  header.setUint32(40, pcm.length, true);
  const bytes = new Uint8Array(44 + pcm.length);
  bytes.set(new Uint8Array(header.buffer), 0);
  bytes.set(pcm, 44);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
