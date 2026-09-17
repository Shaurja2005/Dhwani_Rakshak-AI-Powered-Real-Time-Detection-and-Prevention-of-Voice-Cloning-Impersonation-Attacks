import { test } from "node:test";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";

import { VoiceGuardClient, VoiceGuardError, callMetadata, floatTo16BitPCM, verifyWebhookSignature } from "../src/index.ts";

test("analyzeFile sends auth, metadata and base64 audio", async () => {
  let seen: { url: string; init: RequestInit } | undefined;
  const fakeFetch = (async (url: string, init: RequestInit) => {
    seen = { url, init };
    return new Response(JSON.stringify({ session_risk: { state: "LOW" }, policy_decision: null, agent_prompt: "" }), {
      status: 200,
    });
  }) as unknown as typeof fetch;
  const vg = new VoiceGuardClient({ baseUrl: "https://vg.example/", apiKey: "vg_test_key", fetch: fakeFetch });
  const res = await vg.analyzeFile(new Uint8Array([1, 2, 3]), { tenantId: "bank-a" });
  assert.equal(res.session_risk.state, "LOW");
  assert.equal(seen?.url, "https://vg.example/v1/analyze/file");
  assert.equal((seen?.init.headers as Record<string, string>).Authorization, "Bearer vg_test_key");
  const body = JSON.parse(String(seen?.init.body));
  assert.equal(body.audio_base64, "AQID");
  assert.equal(body.call_metadata.tenant_id, "bank-a");
});

test("HTTP errors surface as VoiceGuardError with status", async () => {
  const fakeFetch = (async () => new Response(JSON.stringify({ detail: "invalid API key" }), { status: 401 })) as unknown as typeof fetch;
  const vg = new VoiceGuardClient({ baseUrl: "https://vg.example", apiKey: "bad", fetch: fakeFetch });
  await assert.rejects(vg.evidence("sha256:x"), (e: unknown) => e instanceof VoiceGuardError && e.status === 401);
});

test("stream opens with call_metadata, sends binary audio and relays events", async () => {
  const sent: unknown[] = [];
  class FakeWS {
    static last: FakeWS;
    url: string;
    readyState = 1;
    binaryType = "";
    onmessage: ((m: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    onopen: (() => void) | null = null;
    onerror: (() => void) | null = null;
    constructor(url: string) {
      this.url = url;
      FakeWS.last = this;
    }
    send(d: unknown) {
      sent.push(d);
      if (typeof d === "string" && d.includes('"close"')) {
        this.onmessage?.({ data: JSON.stringify({ type: "session_risk", data: { state: "LOW" } }) });
        this.onclose?.();
      }
    }
  }
  const events: string[] = [];
  const vg = new VoiceGuardClient({ baseUrl: "https://vg.example", apiKey: "k&1", WebSocket: FakeWS as unknown as typeof WebSocket });
  const s = vg.stream({ tenantId: "bank-a", onEvent: (e) => events.push(e.type) });
  assert.equal(FakeWS.last.url, "wss://vg.example/v1/stream?api_key=k%261");
  await s.open();
  s.sendAudio(new Uint8Array(320));
  await s.close();
  assert.equal(JSON.parse(String(sent[0])).call_metadata.tenant_id, "bank-a");
  assert.ok(sent[1] instanceof Uint8Array);
  assert.deepEqual(events, ["session_risk"]);
});

test("webhook signature verification matches the server's HMAC scheme", async () => {
  const body = '{"type":"policy_decision"}';
  const t = 1_700_000_000;
  const mac = createHmac("sha256", "s3cret").update(`${t}.${body}`).digest("hex");
  const header = `t=${t},v1=${mac}`;
  assert.equal(await verifyWebhookSignature("s3cret", body, header, 300, t + 10), true);
  assert.equal(await verifyWebhookSignature("wrong", body, header, 300, t + 10), false);
  assert.equal(await verifyWebhookSignature("s3cret", body, header, 300, t + 9999), false);
});

test("helpers", () => {
  const m = callMetadata("t", { claimed_identity_id: "cust-1" });
  assert.match(m.session_id, /^[0-9a-f-]{36}$/);
  assert.equal(m.claimed_identity_id, "cust-1");
  assert.deepEqual(Array.from(floatTo16BitPCM(new Float32Array([0, 1, -1]))), [0, 0, 255, 127, 0, 128]);
});
