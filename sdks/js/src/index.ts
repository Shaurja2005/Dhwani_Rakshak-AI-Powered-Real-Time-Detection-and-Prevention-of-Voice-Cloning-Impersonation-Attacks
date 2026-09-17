/**
 * VoiceGuard JS/TS SDK (B12-T06) — browser and Node 20+.
 *
 *   const vg = new VoiceGuardClient({ baseUrl: "https://voiceguard.bank.internal", apiKey });
 *   const result = await vg.analyzeFile(bytes, { tenantId: "bank-a" });
 *
 *   const stream = vg.stream({ tenantId: "bank-a", onEvent: (e) => render(e) });
 *   await stream.open();  stream.sendAudio(pcm16le);  await stream.close();
 *
 * Detection is advisory: show results to a human; never auto-decline a customer on them.
 * Uses only erasable TypeScript syntax, so Node can run it directly (type stripping).
 */

export type RiskState = "LOW" | "ELEVATED" | "HIGH" | "ABSTAIN";

export interface CallMetadata {
  session_id: string;
  tenant_id: string;
  direction: "inbound" | "outbound";
  started_at: string;
  channel: "pstn" | "voip" | "webrtc" | "mobile" | "file";
  codec_hint: string;
  source_sample_rate: number;
  consent_basis: "legitimate_use" | "explicit_consent" | "none";
  caller_number?: string;
  claimed_identity_id?: string;
  language_hint?: string;
}

export interface SessionRisk {
  session_id: string;
  updated_at: string;
  risk_score: number;
  state: RiskState;
  p_spoof_session_max: number;
  p_spoof_session_mean: number;
  drivers: { factor: string; weight: number; detail: string }[];
  timeline: { window_id: number; p_spoof: number; state: RiskState }[];
  model_versions: Record<string, string>;
  abstain_ratio: number;
}

export interface PolicyDecision {
  session_id: string;
  decided_at: string;
  threshold_profile: string;
  actions: string[];
  rationale: string;
  evidence_bundle_id: string;
  shadow_mode: boolean;
  suppressed: boolean;
}

export type VoiceGuardEvent =
  | { type: "session_started"; data: { session_id: string } }
  | { type: "window_score"; data: Record<string, unknown> }
  | { type: "session_risk"; data: SessionRisk }
  | { type: "context_signals"; data: Record<string, unknown> }
  | { type: "policy_decision"; data: PolicyDecision }
  | { type: "error"; detail: string };

export interface ClientOptions {
  baseUrl: string;
  apiKey: string;
  fetch?: typeof fetch;
  WebSocket?: typeof WebSocket;
}

export class VoiceGuardError extends Error {
  status: number;
  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`);
    this.status = status;
  }
}

function uuid(): string {
  return globalThis.crypto.randomUUID();
}

function toBase64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

export function callMetadata(tenantId: string, extra: Partial<CallMetadata> = {}): CallMetadata {
  return {
    session_id: uuid(),
    tenant_id: tenantId,
    direction: "inbound",
    started_at: new Date().toISOString(),
    channel: "webrtc",
    codec_hint: "pcm",
    source_sample_rate: 16000,
    consent_basis: "legitimate_use",
    ...extra,
  };
}

export class VoiceGuardClient {
  private baseUrl: string;
  private apiKey: string;
  private fetchImpl: typeof fetch;
  private WS: typeof WebSocket | undefined;

  constructor(opts: ClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/$/, "");
    this.apiKey = opts.apiKey;
    this.fetchImpl = opts.fetch ?? globalThis.fetch.bind(globalThis);
    this.WS = opts.WebSocket ?? (globalThis as { WebSocket?: typeof WebSocket }).WebSocket;
  }

  async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await this.fetchImpl(this.baseUrl + path, {
      method,
      headers: { Authorization: `Bearer ${this.apiKey}`, "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await res.text();
    const json = text ? JSON.parse(text) : undefined;
    if (!res.ok) throw new VoiceGuardError(res.status, json?.detail ?? text);
    return json as T;
  }

  analyzeFile(audio: Uint8Array, opts: { tenantId: string; metadata?: Partial<CallMetadata> }) {
    return this.request<{ session_risk: SessionRisk; policy_decision: PolicyDecision | null; agent_prompt: string }>(
      "POST",
      "/v1/analyze/file",
      {
        call_metadata: callMetadata(opts.tenantId, { channel: "file", ...opts.metadata }),
        audio_base64: toBase64(audio),
      },
    );
  }

  evidence(bundleId: string) {
    return this.request<{ bundle: Record<string, unknown>; verified: boolean }>("GET", `/v1/evidence/${bundleId}`);
  }

  feedback(bundleId: string, label: "true_positive" | "false_positive" | "unsure", analyst: string, note = "") {
    return this.request("POST", `/v1/evidence/${bundleId}/feedback`, { label, analyst, note });
  }

  stream(opts: {
    tenantId: string;
    onEvent: (e: VoiceGuardEvent) => void;
    metadata?: Partial<CallMetadata>;
    encoding?: "pcm_s16le" | "mulaw" | "alaw";
    sampleRate?: number;
  }): VoiceGuardStream {
    if (!this.WS) throw new Error("WebSocket is not available; pass options.WebSocket");
    const wsUrl = this.baseUrl.replace(/^http/, "ws") + `/v1/stream?api_key=${encodeURIComponent(this.apiKey)}`;
    return new VoiceGuardStream(new this.WS(wsUrl), opts, callMetadata(opts.tenantId, opts.metadata));
  }
}

export class VoiceGuardStream {
  readonly metadata: CallMetadata;
  private ws: WebSocket;
  private opts: { onEvent: (e: VoiceGuardEvent) => void; encoding?: string; sampleRate?: number };
  private closed: Promise<void>;

  constructor(ws: WebSocket, opts: { onEvent: (e: VoiceGuardEvent) => void; encoding?: string; sampleRate?: number },
              metadata: CallMetadata) {
    this.ws = ws;
    this.opts = opts;
    this.metadata = metadata;
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (m: MessageEvent) => this.opts.onEvent(JSON.parse(String(m.data)) as VoiceGuardEvent);
    this.closed = new Promise((resolve) => {
      this.ws.onclose = () => resolve();
    });
  }

  open(): Promise<void> {
    return new Promise((resolve, reject) => {
      const start = () => {
        this.ws.send(JSON.stringify({
          call_metadata: this.metadata,
          encoding: this.opts.encoding ?? "pcm_s16le",
          sample_rate: this.opts.sampleRate ?? 16000,
        }));
        resolve();
      };
      if (this.ws.readyState === 1) start();
      else {
        this.ws.onopen = start;
        this.ws.onerror = () => reject(new Error("VoiceGuard stream failed to connect"));
      }
    });
  }

  sendAudio(chunk: Uint8Array | ArrayBuffer): void {
    this.ws.send(chunk);
  }

  sendTranscript(segments: { start_ms: number; end_ms: number; text: string }[]): void {
    this.ws.send(JSON.stringify({ type: "transcript", segments }));
  }

  close(): Promise<void> {
    this.ws.send(JSON.stringify({ type: "close" }));
    return this.closed;
  }
}

/** Convert Float32 samples [-1, 1] (e.g. from an AudioWorklet) to 16-bit little-endian PCM bytes. */
export function floatTo16BitPCM(samples: Float32Array): Uint8Array {
  const out = new DataView(new ArrayBuffer(samples.length * 2));
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(out.buffer);
}

/** Verify an X-VoiceGuard-Signature header (t=<unix>,v1=<hex HMAC-SHA256 of "t.body">). Works in Node and browsers. */
export async function verifyWebhookSignature(secret: string, body: string, header: string, toleranceS = 300,
                                             nowS = Date.now() / 1000): Promise<boolean> {
  const parts = Object.fromEntries(header.split(",").map((p) => p.split("=", 2) as [string, string]));
  const t = Number(parts.t);
  if (!Number.isFinite(t) || Math.abs(nowS - t) > toleranceS || !parts.v1) return false;
  const enc = new TextEncoder();
  const key = await globalThis.crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" },
                                                        false, ["sign"]);
  const sig = new Uint8Array(await globalThis.crypto.subtle.sign("HMAC", key, enc.encode(`${t}.${body}`)));
  const hex = Array.from(sig, (b) => b.toString(16).padStart(2, "0")).join("");
  if (hex.length !== parts.v1.length) return false;
  let diff = 0;
  for (let i = 0; i < hex.length; i++) diff |= hex.charCodeAt(i) ^ parts.v1.charCodeAt(i);
  return diff === 0;
}
