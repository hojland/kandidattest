# Hybrid Provider Architecture

## Problem

Qwen3-0.6B is too small to follow structured interview prompts. It dumps markup literally, leaks question IDs, and triggers match prematurely. Users need an option between local WebGPU (free, private) and API-based (better quality, requires key).

## Design

### Provider Selection

Shown before storkreds selection. Two modes:

1. **Local (WebGPU)** — Qwen3-1.7B-ONNX q4f16 (~1.4GB, cached after first download)
2. **API (OpenAI-compatible)** — user provides base URL, API key, model name

API presets for common providers: OpenAI, Groq, Together, Ollama (localhost).

### API Key Security

- **Default: in-memory only** — key stored in React state, gone when tab closes
- **Optional: sessionStorage** — checkbox "Husk nøgle for denne session", survives page reloads but not browser close
- Never persisted to localStorage or disk
- Clear warning that key is stored client-side

### Two Prompt Tiers

**Local (1.7B):** Simplified prompt. No `[VALG:]` markup. Natural conversation, shorter instructions. Follow-ups are free-text only. Fewer questions required (10 national + 2 local).

**API (smart model):** Full structured prompt with `[VALG:]` button support, thematic grouping, detailed follow-up protocol. 15 national + 3 local questions.

### Auto-Start Conversation

After storkreds selection + model ready, the app automatically appends a hidden user message to trigger the model's welcome + first question. The user sees only the assistant's opening message.

### File Changes

- **New:** `src/components/ProviderSelector.tsx` — provider choice UI with API config
- **New:** `src/adapters/api-adapter.ts` — OpenAI-compatible streaming adapter
- **Rename:** `src/adapters/chat-model-adapter.ts` → `src/adapters/local-adapter.ts`
- **Modified:** `src/lib/system-prompt.ts` — two exports: `buildLocalPrompt()`, `buildApiPrompt()`
- **Modified:** `src/workers/llm-worker.ts` — model ID changed to Qwen3-1.7B-ONNX
- **Modified:** `src/App.tsx` — provider selection flow, conditional adapter, auto-start

### Provider Type

```typescript
type Provider =
  | { type: "local" }
  | { type: "api"; baseUrl: string; apiKey: string; model: string };
```
