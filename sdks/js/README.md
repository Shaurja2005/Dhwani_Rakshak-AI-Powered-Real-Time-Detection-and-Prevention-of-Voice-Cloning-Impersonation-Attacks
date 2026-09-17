# @voiceguard/sdk

Browser + Node (>= 20) SDK for the VoiceGuard API. Detection is **advisory**:
show results to a human; never auto-decline a customer on them.

```ts
import { VoiceGuardClient, floatTo16BitPCM } from "@voiceguard/sdk";

const vg = new VoiceGuardClient({ baseUrl: "https://voiceguard.bank.internal", apiKey });

// Post-call
const result = await vg.analyzeFile(wavBytes, { tenantId: "bank-a" });

// Live: stream 16 kHz mono PCM, render events
const stream = vg.stream({ tenantId: "bank-a", onEvent: (e) => {
  if (e.type === "policy_decision" && !e.data.suppressed) showBanner(e.data);
}});
await stream.open();
stream.sendAudio(pcmBytes);
await stream.close();
```

## Microphone / collaboration-platform capture (browser)

```ts
const ctx = new AudioContext({ sampleRate: 16000 });
const media = await navigator.mediaDevices.getUserMedia({ audio: true });
await ctx.audioWorklet.addModule("pcm-worklet.js");   // posts Float32Array frames
const node = new AudioWorkletNode(ctx, "pcm-worklet");
node.port.onmessage = (ev) => stream.sendAudio(floatTo16BitPCM(ev.data));
ctx.createMediaStreamSource(media).connect(node);
```

For browsers, prefer a short-lived, `stream`-scoped key issued by your backend
(`POST /v1/tenants/{t}/keys`) — never embed an admin key in a web page.

## Webhooks

```ts
import { verifyWebhookSignature } from "@voiceguard/sdk";
const ok = await verifyWebhookSignature(secret, rawBody, req.headers["x-voiceguard-signature"]);
```

## Develop

```bash
npm install && npm test && npm run build
```
