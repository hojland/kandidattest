# Hybrid Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let users choose between local WebGPU (Qwen3-1.7B) and OpenAI-compatible API for the chat LLM, with auto-start conversation and proper API key handling.

**Architecture:** Provider selection screen shown first. Two adapter implementations share the same assistant-ui runtime. API keys held in React state (in-memory) with optional sessionStorage persistence. System prompt has two tiers: simplified for local, full structured for API.

**Tech Stack:** assistant-ui LocalRuntime, Transformers.js WebGPU, OpenAI-compatible streaming API (fetch + SSE), React state + sessionStorage

---

### Task 1: Provider type and shared config

**Files:**
- Create: `src/lib/provider.ts`

**Step 1: Create provider types**

```typescript
export type Provider =
  | { type: "local" }
  | { type: "api"; baseUrl: string; apiKey: string; model: string };

export const API_PRESETS: { label: string; baseUrl: string; defaultModel: string }[] = [
  { label: "OpenAI", baseUrl: "https://api.openai.com/v1", defaultModel: "gpt-4o-mini" },
  { label: "Groq", baseUrl: "https://api.groq.com/openai/v1", defaultModel: "llama-3.3-70b-versatile" },
  { label: "Together", baseUrl: "https://api.together.xyz/v1", defaultModel: "meta-llama/Llama-3.3-70B-Instruct-Turbo" },
  { label: "Ollama", baseUrl: "http://localhost:11434/v1", defaultModel: "llama3.2" },
];

/** Store API key in sessionStorage (opt-in). Never localStorage. */
export function storeApiSession(provider: Provider): void {
  if (provider.type !== "api") return;
  sessionStorage.setItem("ktest_provider", JSON.stringify({
    baseUrl: provider.baseUrl,
    apiKey: provider.apiKey,
    model: provider.model,
  }));
}

/** Retrieve API config from sessionStorage, or null. */
export function loadApiSession(): Provider | null {
  const raw = sessionStorage.getItem("ktest_provider");
  if (!raw) return null;
  try {
    const { baseUrl, apiKey, model } = JSON.parse(raw);
    if (baseUrl && apiKey && model) return { type: "api", baseUrl, apiKey, model };
  } catch { /* ignore */ }
  return null;
}

/** Clear stored session. */
export function clearApiSession(): void {
  sessionStorage.removeItem("ktest_provider");
}
```

**Step 2: Verify file compiles**

Run: `npx vite build 2>&1 | tail -5`
Expected: Build succeeds

---

### Task 2: Provider selection UI

**Files:**
- Create: `src/components/ProviderSelector.tsx`

**Step 1: Build provider selector component**

```tsx
import { useState, useEffect } from "react";
import type { Provider } from "../lib/provider";
import { API_PRESETS, loadApiSession } from "../lib/provider";

interface Props {
  webgpuSupported: boolean;
  onSelect: (provider: Provider) => void;
}

export function ProviderSelector({ webgpuSupported, onSelect }: Props) {
  const [mode, setMode] = useState<"local" | "api" | null>(null);
  const [preset, setPreset] = useState(0);
  const [baseUrl, setBaseUrl] = useState(API_PRESETS[0].baseUrl);
  const [model, setModel] = useState(API_PRESETS[0].defaultModel);
  const [apiKey, setApiKey] = useState("");
  const [remember, setRemember] = useState(false);

  // Restore from sessionStorage on mount
  useEffect(() => {
    const saved = loadApiSession();
    if (saved && saved.type === "api") {
      setMode("api");
      setBaseUrl(saved.baseUrl);
      setApiKey(saved.apiKey);
      setModel(saved.model);
      setRemember(true);
    }
  }, []);

  const handlePresetChange = (idx: number) => {
    setPreset(idx);
    setBaseUrl(API_PRESETS[idx].baseUrl);
    setModel(API_PRESETS[idx].defaultModel);
  };

  if (mode === null) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 gap-6">
        <h1 className="text-2xl font-bold text-red-700">Kandidattest</h1>
        <p className="text-gray-600 text-center max-w-sm">
          Vælg hvordan AI-modellen skal køre
        </p>
        <div className="flex flex-col gap-3 w-full max-w-sm">
          <button
            onClick={() => setMode("local")}
            disabled={!webgpuSupported}
            className="p-4 border rounded-xl text-left hover:border-red-300 hover:bg-red-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <div className="font-medium">Lokal (WebGPU)</div>
            <div className="text-sm text-gray-500 mt-1">
              Qwen3 1.7B — kører i browseren, ~1.4 GB download
              {!webgpuSupported && " (ikke understøttet i denne browser)"}
            </div>
          </button>
          <button
            onClick={() => setMode("api")}
            className="p-4 border rounded-xl text-left hover:border-red-300 hover:bg-red-50 transition-colors"
          >
            <div className="font-medium">API (OpenAI-kompatibel)</div>
            <div className="text-sm text-gray-500 mt-1">
              Brug din egen API-nøgle — OpenAI, Groq, Together, Ollama m.fl.
            </div>
          </button>
        </div>
      </div>
    );
  }

  if (mode === "local") {
    onSelect({ type: "local" });
    return null;
  }

  // API config form
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-sm space-y-4">
        <h2 className="text-lg font-bold text-red-700">API-opsætning</h2>

        <div>
          <label className="text-sm font-medium text-gray-700">Udbyder</label>
          <select
            value={preset}
            onChange={(e) => handlePresetChange(Number(e.target.value))}
            className="mt-1 w-full border rounded-lg px-3 py-2 text-sm"
          >
            {API_PRESETS.map((p, i) => (
              <option key={p.label} value={i}>{p.label}</option>
            ))}
            <option value={-1}>Anden...</option>
          </select>
        </div>

        <div>
          <label className="text-sm font-medium text-gray-700">Base URL</label>
          <input
            type="url"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            className="mt-1 w-full border rounded-lg px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="text-sm font-medium text-gray-700">Model</label>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="mt-1 w-full border rounded-lg px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="text-sm font-medium text-gray-700">API-nøgle</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-..."
            className="mt-1 w-full border rounded-lg px-3 py-2 text-sm"
          />
        </div>

        <label className="flex items-center gap-2 text-sm text-gray-600">
          <input
            type="checkbox"
            checked={remember}
            onChange={(e) => setRemember(e.target.checked)}
          />
          Husk for denne session (sessionStorage)
        </label>

        <p className="text-xs text-gray-400">
          Din nøgle gemmes kun i hukommelsen{remember ? " og sessionStorage (ryddes når browseren lukkes)" : ""} — aldrig på disk.
        </p>

        <div className="flex gap-2">
          <button
            onClick={() => setMode(null)}
            className="px-4 py-2 border rounded-lg text-sm"
          >
            Tilbage
          </button>
          <button
            onClick={() => {
              if (!apiKey || !baseUrl || !model) return;
              onSelect({ type: "api", baseUrl, apiKey, model });
            }}
            disabled={!apiKey || !baseUrl || !model}
            className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 disabled:opacity-50"
          >
            Fortsæt
          </button>
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Verify build**

Run: `npx vite build 2>&1 | tail -5`
Expected: Build succeeds (component not yet wired up but should compile)

---

### Task 3: OpenAI-compatible API adapter

**Files:**
- Create: `src/adapters/api-adapter.ts`

**Step 1: Create streaming API adapter**

This adapter calls any OpenAI-compatible `/chat/completions` endpoint with `stream: true` and parses SSE chunks. It reuses `stripThinking` and `parseChoices` from the local adapter.

```typescript
import type { ChatModelAdapter } from "@assistant-ui/react";
import { stripThinking, parseChoices, nextChoiceId } from "./shared";

export function createApiAdapter(
  baseUrl: string,
  apiKey: string,
  model: string,
  systemPrompt: string,
): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const apiMessages = [
        ...(systemPrompt ? [{ role: "system" as const, content: systemPrompt }] : []),
        ...messages.map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content
            .filter((p): p is { type: "text"; text: string } => p.type === "text")
            .map((p) => p.text)
            .join(""),
        })),
      ];

      const response = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: apiMessages,
          stream: true,
          temperature: 0.7,
          max_tokens: 1024,
        }),
        signal: abortSignal,
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`API error ${response.status}: ${err}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let fullText = "";
      let buffer = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") continue;

            try {
              const chunk = JSON.parse(data);
              const delta = chunk.choices?.[0]?.delta?.content;
              if (!delta) continue;

              fullText += delta;
              const { text, isThinking } = stripThinking(fullText);

              if (isThinking && !text) {
                yield { content: [{ type: "text" as const, text: "💭 Tænker..." }] };
                continue;
              }

              const displayText = isThinking ? text + "\n\n💭 Tænker..." : text;
              const { textBefore, options, hasCompleteChoice, hasPartialChoice } =
                parseChoices(displayText);

              if (hasPartialChoice) {
                const content: Array<{ type: "text"; text: string }> = [];
                if (textBefore) content.push({ type: "text" as const, text: textBefore });
                content.push({ type: "text" as const, text: "💭 Tænker..." });
                yield { content };
              } else if (hasCompleteChoice && options.length > 0) {
                const toolCallId = nextChoiceId();
                const content: Array<any> = [];
                if (textBefore) content.push({ type: "text" as const, text: textBefore });
                content.push({
                  type: "tool-call" as const,
                  toolCallId,
                  toolName: "user_choice",
                  args: { options },
                  argsText: JSON.stringify({ options }),
                });
                yield {
                  content,
                  status: { type: "requires-action" as const, reason: "tool-calls" as const },
                };
              } else {
                yield { content: [{ type: "text" as const, text: displayText }] };
              }
            } catch { /* skip malformed chunks */ }
          }
        }
      } finally {
        reader.releaseLock();
      }
    },
  };
}
```

**Step 2: Verify build**

Run: `npx vite build 2>&1 | tail -5`
Expected: Build succeeds

---

### Task 4: Extract shared adapter utilities

**Files:**
- Create: `src/adapters/shared.ts`
- Modify: `src/adapters/chat-model-adapter.ts` — import from shared instead of defining locally

**Step 1: Create shared module**

Extract `stripThinking`, `parseChoices`, and `choiceCounter` into `src/adapters/shared.ts`:

```typescript
/**
 * Shared utilities for both local and API adapters.
 */

/** Strip <think>...</think> blocks from streamed text. */
export function stripThinking(raw: string): { text: string; isThinking: boolean } {
  let text = raw.replace(/<think>[\s\S]*?<\/think>/g, "");
  const isThinking = /<think>(?![\s\S]*<\/think>)/.test(raw);
  if (isThinking) {
    text = text.replace(/<think>[\s\S]*$/, "");
  }
  return { text: text.trim(), isThinking };
}

/** Parse [VALG: opt1 | opt2 | opt3] markup from LLM output. */
export function parseChoices(text: string): {
  textBefore: string;
  options: string[];
  hasCompleteChoice: boolean;
  hasPartialChoice: boolean;
} {
  const completeMatch = text.match(/^([\s\S]*?)\[VALG:\s*(.+?)\]\s*$/);
  if (completeMatch) {
    const textBefore = completeMatch[1].trim();
    const options = completeMatch[2].split("|").map((o) => o.trim()).filter(Boolean);
    return { textBefore, options, hasCompleteChoice: true, hasPartialChoice: false };
  }

  const partialMatch = text.match(/^([\s\S]*?)\[VALG:[^\]]*$/);
  if (partialMatch) {
    return { textBefore: partialMatch[1].trim(), options: [], hasCompleteChoice: false, hasPartialChoice: true };
  }

  return { textBefore: text, options: [], hasCompleteChoice: false, hasPartialChoice: false };
}

let choiceCounter = 0;
export function nextChoiceId(): string {
  return `choice_${++choiceCounter}`;
}
```

**Step 2: Update `chat-model-adapter.ts`**

Remove the local `stripThinking`, `parseChoices`, and `choiceCounter` definitions. Replace with:

```typescript
import { stripThinking, parseChoices, nextChoiceId } from "./shared";
```

And replace `choice_${++choiceCounter}` with `nextChoiceId()` on line 134.

**Step 3: Verify build**

Run: `npx vite build 2>&1 | tail -5`
Expected: Build succeeds

---

### Task 5: Two-tier system prompt

**Files:**
- Modify: `src/lib/system-prompt.ts`

**Step 1: Add `buildLocalPrompt` export**

Keep the existing `buildSystemPrompt` (rename to `buildApiPrompt`) and add a much simpler `buildLocalPrompt` for the 1.7B model. The local prompt should:
- Be much shorter (fits in small context)
- No `[VALG:]` markup instructions
- No question IDs
- Just list questions as plain text
- Require fewer questions (10 national + 2 local)

Add at the top of the file, after `THEME_ORDER`:

```typescript
export function buildLocalPrompt(
  nationalQuestions: Record<string, string>,
  localQuestions: Record<string, string>,
  storkredsName: string,
): string {
  const questions = Object.values(nationalQuestions)
    .map((q, i) => `${i + 1}. ${q}`)
    .join("\n");

  const local = Object.values(localQuestions)
    .map((q, i) => `${i + 1}. ${q}`)
    .join("\n");

  return `Du er en venlig politisk rådgiver. Du hjælper vælgere med at finde deres kandidat til FV2026.
Svar KUN på dansk. Hold beskeder korte — maks 2 sætninger.

Spørgsmål du skal stille (ét ad gangen, med dine egne ord):
${questions}

Lokale spørgsmål for ${storkredsName}:
${local}

Regler:
- Stil ét spørgsmål ad gangen med dine egne ord
- Dæk mindst 10 nationale og 2 lokale spørgsmål
- Hvis svaret er uklart, stil ét opfølgende spørgsmål
- Vær neutral — vis aldrig din holdning
- Når du er færdig, skriv: [KLAR TIL MATCH]
- Start med en kort velkomst og dit første spørgsmål`;
}
```

**Step 2: Rename `buildSystemPrompt` to `buildApiPrompt`**

Rename the existing function. Update the export name.

**Step 3: Verify build**

Run: `npx vite build 2>&1 | tail -5`
Expected: Build succeeds

---

### Task 6: Upgrade local model to Qwen3-1.7B

**Files:**
- Modify: `src/workers/llm-worker.ts:10`

**Step 1: Change model ID**

Change line 10 from:
```typescript
this.instance = await pipeline("text-generation", "onnx-community/Qwen3-0.6B-ONNX", {
```
to:
```typescript
this.instance = await pipeline("text-generation", "onnx-community/Qwen3-1.7B-ONNX", {
```

**Step 2: Verify build**

Run: `npx vite build 2>&1 | tail -5`
Expected: Build succeeds

---

### Task 7: Rewire App.tsx for provider flow + auto-start

**Files:**
- Modify: `src/App.tsx`

This is the biggest task. The app flow becomes:

1. Provider selection (new first screen)
2. Storkreds selection (existing)
3. Model loading (only for local provider)
4. Chat with auto-started first message

**Step 1: Update imports**

Replace:
```typescript
import { createLLMAdapter } from "./adapters/chat-model-adapter";
import { buildSystemPrompt } from "./lib/system-prompt";
```
with:
```typescript
import { createLLMAdapter } from "./adapters/chat-model-adapter";
import { createApiAdapter } from "./adapters/api-adapter";
import { buildLocalPrompt, buildApiPrompt } from "./lib/system-prompt";
import type { Provider } from "./lib/provider";
import { storeApiSession } from "./lib/provider";
import { ProviderSelector } from "./components/ProviderSelector";
```

**Step 2: Add provider state**

Add to the state declarations in `App()`:
```typescript
const [provider, setProvider] = useState<Provider | null>(null);
```

**Step 3: Handle provider selection**

Add a handler that stores to sessionStorage if the user opted in:
```typescript
const handleProviderSelect = useCallback((p: Provider) => {
  setProvider(p);
}, []);
```

**Step 4: Conditional model loading**

The existing `useEffect` that calls `checkWebGPU` and loads models should only run for local provider. For API provider, skip WebGPU check and LLM worker entirely — only load embeddings and data.

Wrap the LLM worker creation in a condition:
```typescript
// Only create LLM worker for local provider
if (provider?.type === "local" && !llmWorkerRef.current) {
  llmWorkerRef.current = new Worker(...);
}
```

Split the `loadModels` effect: embeddings always load, LLM worker only for local.

**Step 5: Conditional adapter creation**

Replace the fixed adapter with:
```typescript
const adapter = provider?.type === "api"
  ? createApiAdapter(provider.baseUrl, provider.apiKey, provider.model, systemPrompt ?? "")
  : createLLMAdapter(llmWorkerRef.current!, systemPrompt ?? "");
```

**Step 6: Conditional prompt builder**

In the system prompt effect, use the right builder:
```typescript
const promptBuilder = provider?.type === "api" ? buildApiPrompt : buildLocalPrompt;
setSystemPrompt(promptBuilder(national, localForStorkreds, selectedStorkreds!));
```

Also handle `storeApiSession` when provider is API and remember is checked (pass through from ProviderSelector).

**Step 7: Auto-start conversation**

Add an effect that triggers the first message once everything is ready:
```typescript
const [conversationStarted, setConversationStarted] = useState(false);

// Auto-start: send hidden trigger once ready
useEffect(() => {
  if (!modelsReady || !systemPrompt || conversationStarted) return;
  // For API, modelsReady is set when embeddings are loaded (no LLM worker needed)
  setConversationStarted(true);
}, [modelsReady, systemPrompt, conversationStarted]);
```

Then, inside the `AssistantRuntimeProvider`, use a small component:
```typescript
function AutoStart({ ready }: { ready: boolean }) {
  const threadRuntime = useThreadRuntime();
  const started = useRef(false);

  useEffect(() => {
    if (!ready || started.current) return;
    started.current = true;
    threadRuntime.append({
      role: "user",
      content: [{ type: "text", text: "Hej! Start venligst samtalen." }],
    });
  }, [ready, threadRuntime]);

  return null;
}
```

**Step 8: Add provider selection screen**

Before the storkreds screen, show provider selection:
```typescript
if (!provider) {
  return (
    <div className="h-dvh flex flex-col max-w-2xl mx-auto">
      <ProviderSelector
        webgpuSupported={!webgpuError}
        onSelect={handleProviderSelect}
      />
    </div>
  );
}
```

**Step 9: Skip loading overlay for API provider**

Only show `ModelLoadingOverlay` for local provider. For API, set `modelsReady` once embeddings finish (no LLM worker to wait for).

**Step 10: Verify build**

Run: `npx vite build 2>&1 | tail -5`
Expected: Build succeeds

---

### Task 8: Smoke test and fix

**Step 1: Run dev server**

Run: `npm run dev`

**Step 2: Test API flow**

- Select "API" → enter a valid OpenAI key → select storkreds
- Verify: chat auto-starts, model asks first question in Danish
- Verify: `[VALG:]` buttons render when model uses them
- Verify: clicking a button sends the choice
- Verify: no API key visible in DOM or localStorage (check devtools Application tab)

**Step 3: Test local flow**

- Select "Local (WebGPU)" → select storkreds
- Verify: model downloads 1.7B (progress overlay shows)
- Verify: chat auto-starts after download
- Verify: conversation is coherent (much better than 0.6B)

**Step 4: Test sessionStorage**

- Select API with "remember" checked → reload page
- Verify: API config is restored from sessionStorage
- Close browser tab → reopen → verify config is gone

**Step 5: Fix any issues found**

---

### Task 9: Commit

```bash
git add src/lib/provider.ts src/adapters/shared.ts src/adapters/api-adapter.ts src/components/ProviderSelector.tsx
git add src/adapters/chat-model-adapter.ts src/lib/system-prompt.ts src/workers/llm-worker.ts src/App.tsx
git commit -m "feat: hybrid provider — local WebGPU or OpenAI-compatible API

- Add provider selection screen (local Qwen3-1.7B vs API)
- Add OpenAI-compatible streaming adapter
- API key stored in-memory by default, optional sessionStorage
- Two prompt tiers: simplified for local, full structured for API
- Auto-start conversation after setup
- Upgrade local model from 0.6B to 1.7B"
```
