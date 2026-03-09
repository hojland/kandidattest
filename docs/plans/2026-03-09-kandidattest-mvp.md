# Kandidattest MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully client-side Danish election candidate matcher that uses a conversational chat interface powered by a small quantized LLM (Qwen3-0.6B via Transformers.js with WebGPU), matching users to candidates via pre-computed embeddings.

**Architecture:** React app (Vite) with zero backend. A small LLM runs in-browser via Transformers.js WebGPU. The chat UI is powered by `assistant-ui` with its `LocalRuntime` — a production-grade chat component library that handles streaming, markdown, auto-scroll, and mobile layout. User responses are embedded client-side with EmbeddingGemma and compared against pre-computed candidate embeddings using cosine similarity.

**Tech Stack:**
- Vite + React + TypeScript
- `assistant-ui` (`@assistant-ui/react`) with `LocalRuntime` for chat UI — provides streaming rendering, markdown, auto-scroll, retry/regenerate, mobile layout
- `shadcn/ui` + Tailwind CSS for styling
- Transformers.js v3 with WebGPU backend
- Chat LLM: `onnx-community/Qwen3-0.6B` (ONNX q4f16, ~400MB in-browser)
- Embedding model: `onnx-community/embeddinggemma-300m-ONNX` (q4f16, ~175MB in-browser) — MTEB Multilingual score 61.15, supports 100+ languages including Danish, MRL dims (768/512/256)
- Pre-computed embeddings generated once via Python with `google/embeddinggemma-300m`
- Deployed as static files (GitHub Pages / Cloudflare Pages)

---

## Data we have

- `data/candidates_raw.json` — 918 candidates, 24 national questions (answer scale -2 to +2), free-text comments (689 candidates), priorities, local questions
- Question keys: `tv2-fv26-danmark-1` through `tv2-fv26-danmark-24`
- **Missing:** Question text strings (only keys exist in data). Must be scraped from the rendered TV2 page.

## Architecture overview

```
┌──────────────────────────────────────────────────────┐
│  Browser (all client-side)                           │
│                                                      │
│  ┌────────────────┐    ┌────────────────────┐        │
│  │ assistant-ui   │    │ Web Worker:         │        │
│  │ LocalRuntime   │◄──►│ Qwen3-0.6B (WebGPU)│        │
│  │ + Thread UI    │    │ via Transformers.js │        │
│  └───────┬────────┘    └────────────────────┘        │
│          │                                           │
│  ┌───────▼────────┐    ┌────────────────────┐        │
│  │ Web Worker:    │    │ Pre-computed        │        │
│  │ EmbeddingGemma │◄──►│ candidate embeddings│        │
│  │ 300M (WebGPU)  │    │ (768d float32 bin)  │        │
│  └────────────────┘    └────────────────────┘        │
│                                                      │
│  Matching: cosine similarity on profile embeddings   │
│  → ranked candidate list with match cards            │
└──────────────────────────────────────────────────────┘
```

## Matching strategy (Phase 1 — MVP)

Keep it simple: single profile embedding comparison.

1. Candidate "profile text" = pitch + answer stances + free-text comments (pre-computed once)
2. LLM converses with user about political topics (guided by 24 question themes)
3. After 5-8 exchanges, user's conversation text is embedded with EmbeddingGemma client-side
4. Cosine similarity against pre-computed candidate embeddings → ranked match list
5. Top 10 matches shown as candidate cards in the chat thread

---

## Task 1: Project scaffolding

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`
- Create: `index.html`, `src/main.tsx`, `src/App.tsx`
- Create: `tailwind.config.ts`, `postcss.config.js`
- Create: `.gitignore`

**Step 1: Initialize React project with Vite**

```bash
cd /Users/hojland/kandidattest
npm create vite@latest . -- --template react-ts
```

Accept overwrite prompts for existing files.

**Step 2: Install core dependencies**

```bash
npm install
npm install @huggingface/transformers
npm install @assistant-ui/react
npm install tailwindcss @tailwindcss/vite
```

**Step 3: Configure Vite**

Edit `vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },
  worker: {
    format: "es",
  },
});
```

**Step 4: Set up Tailwind**

In `src/index.css`:
```css
@import "tailwindcss";
```

**Step 5: Update .gitignore**

```
node_modules/
dist/
data/candidates_raw.json
.DS_Store
*.local
```

**Step 6: Verify dev server starts**

```bash
npm run dev
```

Expected: Vite dev server on localhost:5173 with React hello world.

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: scaffold Vite + React + TypeScript + Tailwind + assistant-ui"
```

---

## Task 2: Extract question texts from TV2

**Files:**
- Create: `scripts/scrape_questions.py`
- Create: `data/questions.json`

**Step 1: Write scraper script**

Uses `agent-browser` to navigate to a candidate page that has all national answers and extract the rendered question texts from the DOM. The questions are rendered in the HTML but not in the data-props JSON.

```python
#!/usr/bin/env python3
"""Extract question texts from TV2's kandidattest rendered page.

Strategy: Open a candidate page, snapshot the DOM, and find question text elements.
The exact selectors need to be discovered by inspecting with agent-browser snapshot.
"""
import subprocess
import json
import re
import time

def ab(*args, timeout=30):
    r = subprocess.run(
        ["agent-browser"] + list(args),
        capture_output=True, text=True, timeout=timeout
    )
    return re.sub(r'\x1b\[[0-9;]*m', '', r.stdout.strip())

# Open a candidate page with all national answers (Mette Abildgaard)
ab("open", "https://nyheder.tv2.dk/folketingsvalg/kandidat/109366")
time.sleep(5)

# First, snapshot to discover DOM structure
snapshot = ab("snapshot", "-c")
print(snapshot[:2000])

# Then extract question texts — selectors TBD from snapshot inspection
# The page renders questions as labeled sections with the question text
# and the candidate's answer (-2 to +2 scale indicator)
result = ab("eval", """
(() => {
    const questions = {};
    // Discover question elements — adjust selectors based on snapshot
    document.querySelectorAll('[class*="question"], [class*="Question"]').forEach(el => {
        const text = el.textContent.trim();
        if (text) questions[el.className] = text;
    });
    return JSON.stringify(questions);
})()
""")

print(result)
ab("close")
```

This is a best-effort script — the exact DOM selectors need discovery during implementation by inspecting with `agent-browser snapshot`.

**Step 2: Manually verify and save questions**

Save the final mapping as `data/questions.json`:

```json
{
  "tv2-fv26-danmark-1": "Danmark bør bruge flere penge på forsvar",
  "tv2-fv26-danmark-2": "..."
}
```

**Step 3: Commit**

```bash
git add data/questions.json scripts/scrape_questions.py
git commit -m "feat: extract question texts from TV2 kandidattest"
```

---

## Task 3: Process candidate data

**Files:**
- Create: `scripts/process_candidates.py`
- Create: `data/candidates.json` (with profileText for embedding generation)
- Create: `public/candidates.json` (compact, for frontend display)

**Step 1: Write processing script**

```python
#!/usr/bin/env python3
"""Process raw candidate data into compact format for the web app."""
import json

with open("data/candidates_raw.json") as f:
    raw = json.load(f)

with open("data/questions.json") as f:
    questions = json.load(f)

candidates = []
for c in raw:
    answers = {}
    profile_parts = []

    if c.get("pitch"):
        profile_parts.append(c["pitch"])

    for qkey, qtext in questions.items():
        ans = (c.get("answers") or {}).get(qkey, {})
        score = ans.get("answer")
        comment = ans.get("comment", "")

        if score is not None:
            answers[qkey] = {"score": score, "comment": comment}
            stance = {-2: "Helt uenig", -1: "Uenig", 0: "Neutral", 1: "Enig", 2: "Helt enig"}.get(score, "")
            profile_parts.append(f"{qtext}: {stance}")
            if comment:
                profile_parts.append(comment)

    if not answers:
        continue  # Skip candidates with no national answers

    candidates.append({
        "id": c["id"],
        "name": c["name"],
        "party": c["party"],
        "partyLetter": c["partyLetter"],
        "area": c["area"],
        "age": c.get("age"),
        "occupation": c.get("occupation"),
        "pitch": c.get("pitch", ""),
        "priorities": c.get("priorities", []),
        "answers": answers,
        "profileText": "\n".join(profile_parts),
    })

print(f"Processed {len(candidates)} candidates with answers")

# Full version with profileText (for embedding generation)
with open("data/candidates.json", "w") as f:
    json.dump(candidates, f, ensure_ascii=False)

# Compact version without profileText (for frontend, smaller)
compact = [{k: v for k, v in c.items() if k != "profileText"} for c in candidates]
with open("public/candidates.json", "w") as f:
    json.dump(compact, f, ensure_ascii=False)

print(f"Saved public/candidates.json ({len(compact)} candidates)")
```

**Step 2: Run processing**

```bash
mkdir -p public
python3 scripts/process_candidates.py
```

**Step 3: Commit**

```bash
git add scripts/process_candidates.py data/candidates.json public/candidates.json
git commit -m "feat: process candidate data into compact format"
```

---

## Task 4: Generate candidate embeddings

**Files:**
- Create: `scripts/generate_embeddings.py`
- Create: `public/embeddings.bin` (binary float32 array)
- Create: `public/embedding_index.json` (candidate ID → offset mapping)

**Step 1: Write embedding generation script**

Uses the same EmbeddingGemma model server-side (full precision) to generate candidate embeddings that will be compared against client-side q4f16 embeddings. The embeddings should be compatible because the quantized model approximates the full model.

```python
#!/usr/bin/env python3
"""Generate embeddings for candidate profile texts using EmbeddingGemma-300M.

Uses the full-precision model server-side. The browser will use the q4f16
quantized ONNX version for user text embedding. Cosine similarity between
full and quantized embeddings is high enough for ranking.
"""
import json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "google/embeddinggemma-300m"
EMBEDDING_DIM = 768  # Full dim; could truncate to 256 via MRL if needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

with open("data/candidates.json") as f:
    candidates = json.load(f)

texts = [c["profileText"] for c in candidates]
index_entries = [{"id": c["id"], "name": c["name"], "party": c["party"]} for c in candidates]

print(f"Generating embeddings for {len(texts)} candidates...")

BATCH_SIZE = 32
all_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over non-padding tokens
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    all_embeddings.append(embeddings.numpy())
    print(f"  Batch {i // BATCH_SIZE + 1}/{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}")

embeddings_np = np.concatenate(all_embeddings, axis=0).astype(np.float32)

# Save as flat float32 binary
embeddings_np.tofile("public/embeddings.bin")

# Save index mapping
with open("public/embedding_index.json", "w") as f:
    json.dump({
        "dim": EMBEDDING_DIM,
        "count": len(index_entries),
        "candidates": index_entries,
    }, f, ensure_ascii=False)

size_mb = embeddings_np.nbytes / 1024 / 1024
print(f"Saved {embeddings_np.shape} embeddings ({size_mb:.1f} MB)")
# Expected: ~824 × 768 = ~2.4 MB
```

**Step 2: Install dependencies and run**

```bash
pip install transformers torch numpy
python3 scripts/generate_embeddings.py
```

Expected: ~824 embeddings × 768 dims = ~2.4 MB binary file.

**Step 3: Commit**

```bash
git add scripts/generate_embeddings.py public/embeddings.bin public/embedding_index.json
git commit -m "feat: generate candidate profile embeddings with EmbeddingGemma-300M"
```

---

## Task 5: LLM web worker with streaming

**Files:**
- Create: `src/workers/llm-worker.ts`

**Step 1: Write the LLM web worker**

This worker loads Qwen3-0.6B and streams tokens back via `postMessage`. The `TextStreamer` callback fires on each generated token.

```typescript
// src/workers/llm-worker.ts
import { pipeline, TextStreamer } from "@huggingface/transformers";

type Message = { role: string; content: string };

class LLMPipeline {
  static instance: Awaited<ReturnType<typeof pipeline>> | null = null;

  static async getInstance(progressCallback?: (data: any) => void) {
    if (!this.instance) {
      this.instance = await pipeline("text-generation", "onnx-community/Qwen3-0.6B", {
        dtype: "q4f16",
        device: "webgpu",
        progress_callback: progressCallback,
      });
    }
    return this.instance;
  }
}

self.addEventListener("message", async (event: MessageEvent) => {
  const { type, messages, id } = event.data;

  if (type === "load") {
    await LLMPipeline.getInstance((data) => {
      self.postMessage({ type: "progress", ...data });
    });
    self.postMessage({ type: "ready" });
    return;
  }

  if (type === "generate") {
    const generator = await LLMPipeline.getInstance();

    const streamer = new TextStreamer(generator!.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (token: string) => {
        self.postMessage({ type: "token", token, id });
      },
    });

    await generator!(messages as Message[], {
      max_new_tokens: 512,
      temperature: 0.7,
      do_sample: true,
      streamer,
    });

    self.postMessage({ type: "done", id });
  }
});
```

**Step 2: Commit**

```bash
git add src/workers/llm-worker.ts
git commit -m "feat: add LLM web worker with streaming via TextStreamer"
```

---

## Task 6: Embedding web worker

**Files:**
- Create: `src/workers/embedding-worker.ts`

**Step 1: Write the embedding web worker**

Loads EmbeddingGemma-300M q4f16 in-browser for embedding user text.

```typescript
// src/workers/embedding-worker.ts
import { AutoModel, AutoTokenizer } from "@huggingface/transformers";

let model: any = null;
let tokenizer: any = null;

self.addEventListener("message", async (event: MessageEvent) => {
  const { type, text, id } = event.data;

  if (type === "load") {
    tokenizer = await AutoTokenizer.from_pretrained("onnx-community/embeddinggemma-300m-ONNX");
    model = await AutoModel.from_pretrained("onnx-community/embeddinggemma-300m-ONNX", {
      dtype: "q4f16",
      device: "webgpu",
      progress_callback: (data: any) => {
        self.postMessage({ type: "progress", ...data });
      },
    });
    self.postMessage({ type: "ready" });
    return;
  }

  if (type === "embed" && model && tokenizer) {
    const inputs = await tokenizer(text, { padding: true, truncation: true });
    const { last_hidden_state } = await model(inputs);

    // Mean pooling: average over sequence length (dim 1), get [1, 768]
    // Then L2 normalize
    const data = last_hidden_state.data as Float32Array;
    const [, seqLen, dim] = last_hidden_state.dims;
    const embedding = new Float32Array(dim);

    for (let d = 0; d < dim; d++) {
      let sum = 0;
      for (let s = 0; s < seqLen; s++) {
        sum += data[s * dim + d];
      }
      embedding[d] = sum / seqLen;
    }

    // L2 normalize
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += embedding[d] * embedding[d];
    norm = Math.sqrt(norm);
    for (let d = 0; d < dim; d++) embedding[d] /= norm;

    self.postMessage({ type: "embedding", embedding: Array.from(embedding), id });
  }
});
```

**Step 2: Commit**

```bash
git add src/workers/embedding-worker.ts
git commit -m "feat: add EmbeddingGemma web worker for client-side embedding"
```

---

## Task 7: ChatModelAdapter bridging worker to assistant-ui

**Files:**
- Create: `src/adapters/chat-model-adapter.ts`

**Step 1: Write the ChatModelAdapter**

This bridges the LLM web worker's streaming tokens into assistant-ui's `LocalRuntime` async generator interface.

```typescript
// src/adapters/chat-model-adapter.ts
import type { ChatModelAdapter } from "@assistant-ui/react";

export function createLLMAdapter(worker: Worker): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const history = messages.map((m) => ({
        role: m.role,
        content: m.content
          .filter((p): p is { type: "text"; text: string } => p.type === "text")
          .map((p) => p.text)
          .join(""),
      }));

      const id = crypto.randomUUID();
      let fullText = "";
      let resolve: ((v: IteratorResult<string>) => void) | null = null;
      let done = false;

      const onMessage = (e: MessageEvent) => {
        if (e.data.id !== id) return;
        if (e.data.type === "token") {
          const r = resolve;
          resolve = null;
          r?.({ value: e.data.token, done: false });
        }
        if (e.data.type === "done") {
          done = true;
          resolve?.({ value: "", done: true });
        }
      };

      worker.addEventListener("message", onMessage);
      abortSignal.addEventListener("abort", () => {
        done = true;
        resolve?.({ value: "", done: true });
      });

      worker.postMessage({ type: "generate", messages: history, id });

      try {
        while (!done) {
          const result: IteratorResult<string> = await new Promise((r) => {
            resolve = r;
          });
          if (result.done) break;
          fullText += result.value;
          yield { content: [{ type: "text" as const, text: fullText }] };
        }
      } finally {
        worker.removeEventListener("message", onMessage);
      }
    },
  };
}
```

**Step 2: Commit**

```bash
git add src/adapters/chat-model-adapter.ts
git commit -m "feat: add ChatModelAdapter bridging LLM worker to assistant-ui"
```

---

## Task 8: Embedding manager with cosine similarity

**Files:**
- Create: `src/lib/embeddings.ts`

**Step 1: Write the embeddings manager**

Manages pre-computed candidate embeddings and the client-side embedding worker.

```typescript
// src/lib/embeddings.ts

export interface CandidateMatch {
  id: number;
  name: string;
  party: string;
  score: number;
}

interface EmbeddingIndex {
  dim: number;
  count: number;
  candidates: Array<{ id: number; name: string; party: string }>;
}

export class EmbeddingManager {
  private worker: Worker;
  private candidateEmbeddings: Float32Array | null = null;
  private index: EmbeddingIndex | null = null;

  constructor(private onProgress?: (data: any) => void) {
    this.worker = new Worker(
      new URL("../workers/embedding-worker.ts", import.meta.url),
      { type: "module" }
    );
    this.worker.addEventListener("message", (e) => {
      if (e.data.type === "progress" && this.onProgress) {
        this.onProgress(e.data);
      }
    });
  }

  async load(): Promise<void> {
    const [embBuf, indexData] = await Promise.all([
      fetch("/embeddings.bin").then((r) => r.arrayBuffer()),
      fetch("/embedding_index.json").then((r) => r.json()),
    ]);
    this.candidateEmbeddings = new Float32Array(embBuf);
    this.index = indexData;

    return new Promise((resolve) => {
      const handler = (e: MessageEvent) => {
        if (e.data.type === "ready") {
          this.worker.removeEventListener("message", handler);
          resolve();
        }
      };
      this.worker.addEventListener("message", handler);
      this.worker.postMessage({ type: "load" });
    });
  }

  async embed(text: string): Promise<number[]> {
    const id = crypto.randomUUID();
    return new Promise((resolve) => {
      const handler = (e: MessageEvent) => {
        if (e.data.type === "embedding" && e.data.id === id) {
          this.worker.removeEventListener("message", handler);
          resolve(e.data.embedding);
        }
      };
      this.worker.addEventListener("message", handler);
      this.worker.postMessage({ type: "embed", text, id });
    });
  }

  findMatches(userEmbedding: number[], topK = 10): CandidateMatch[] {
    if (!this.candidateEmbeddings || !this.index) return [];
    const { dim, count, candidates } = this.index;
    const results: CandidateMatch[] = [];

    for (let i = 0; i < count; i++) {
      let dot = 0;
      for (let d = 0; d < dim; d++) {
        dot += userEmbedding[d] * this.candidateEmbeddings[i * dim + d];
      }
      results.push({ ...candidates[i], score: dot });
    }

    return results.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}
```

**Step 2: Commit**

```bash
git add src/lib/embeddings.ts
git commit -m "feat: add EmbeddingManager with cosine similarity matching"
```

---

## Task 9: Main App with assistant-ui Thread

**Files:**
- Create: `src/App.tsx`
- Modify: `src/main.tsx`

**Step 1: Write the App component**

Wires together assistant-ui `LocalRuntime`, model loading with progress, and the chat thread.

```tsx
// src/App.tsx
import { useRef, useState, useEffect } from "react";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  Thread,
} from "@assistant-ui/react";
import { createLLMAdapter } from "./adapters/chat-model-adapter";
import { EmbeddingManager } from "./lib/embeddings";

function ModelLoadingOverlay({ items }: { items: Map<string, number> }) {
  if (items.size === 0) return null;
  return (
    <div className="fixed inset-0 bg-black/60 flex flex-col items-center justify-center z-50">
      <h2 className="text-white text-xl font-bold mb-6">Henter AI-modeller...</h2>
      <p className="text-white/70 text-sm mb-4">Første gang tager det et par minutter. Modellerne caches i browseren.</p>
      {[...items.entries()].map(([file, progress]) => (
        <div key={file} className="w-80 mb-3">
          <p className="text-white/80 text-xs truncate">{file}</p>
          <div className="bg-white/20 rounded-full h-2 mt-1">
            <div
              className="bg-red-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const llmWorkerRef = useRef<Worker | null>(null);
  const embeddingsRef = useRef<EmbeddingManager | null>(null);
  const [progressItems, setProgressItems] = useState<Map<string, number>>(new Map());
  const [modelsReady, setModelsReady] = useState(false);

  // Create LLM worker once
  if (!llmWorkerRef.current) {
    llmWorkerRef.current = new Worker(
      new URL("./workers/llm-worker.ts", import.meta.url),
      { type: "module" }
    );
  }

  // Track download progress from both workers
  const handleProgress = (data: any) => {
    if (data.status === "progress" && data.file) {
      setProgressItems((prev) => {
        const next = new Map(prev);
        next.set(data.file, data.progress ?? 0);
        return next;
      });
    }
    if (data.status === "done" && data.file) {
      setProgressItems((prev) => {
        const next = new Map(prev);
        next.delete(data.file);
        return next;
      });
    }
  };

  useEffect(() => {
    llmWorkerRef.current!.addEventListener("message", (e) => handleProgress(e.data));

    async function loadModels() {
      // Load embedding model + pre-computed embeddings
      embeddingsRef.current = new EmbeddingManager(handleProgress);
      await embeddingsRef.current.load();

      // Load LLM
      await new Promise<void>((resolve) => {
        const handler = (e: MessageEvent) => {
          if (e.data.type === "ready") {
            llmWorkerRef.current!.removeEventListener("message", handler);
            resolve();
          }
        };
        llmWorkerRef.current!.addEventListener("message", handler);
        llmWorkerRef.current!.postMessage({ type: "load" });
      });

      setModelsReady(true);
    }

    loadModels();
  }, []);

  const adapter = createLLMAdapter(llmWorkerRef.current);
  const runtime = useLocalRuntime(adapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ModelLoadingOverlay items={progressItems} />
      <div className="h-dvh flex flex-col max-w-2xl mx-auto">
        <header className="px-4 py-3 border-b">
          <h1 className="text-xl font-bold text-red-700">Kandidattest</h1>
          <p className="text-sm text-gray-500">Chat med AI om politik — find din kandidat</p>
        </header>
        <div className="flex-1 overflow-hidden">
          <Thread />
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}
```

**Step 2: Wire up main.tsx**

```tsx
// src/main.tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

**Step 3: Commit**

```bash
git add src/App.tsx src/main.tsx
git commit -m "feat: wire App with assistant-ui Thread, model loading, and progress overlay"
```

---

## Task 10: System prompt and conversation flow

**Files:**
- Create: `src/lib/system-prompt.ts`
- Create: `public/questions.json` (copy from data/)

**Step 1: Write the system prompt builder**

The system prompt instructs Qwen3-0.6B to conduct a natural Danish conversation about the 24 political topics and signal when it has enough information.

```typescript
// src/lib/system-prompt.ts

export function buildSystemPrompt(questions: Record<string, string>): string {
  const questionList = Object.values(questions)
    .map((q, i) => `${i + 1}. ${q}`)
    .join("\n");

  return `Du er en venlig politisk rådgiver der hjælper danske vælgere med at finde deres kandidat til folketingsvalget 2026.

Du skal føre en naturlig samtale på dansk om politiske emner. Dæk disse emner gennem samtalen:
${questionList}

Regler:
- Stil ét spørgsmål ad gangen, i en naturlig rækkefølge
- Brug et venligt, uformelt sprog — som en ven der spørger om dine holdninger
- Opsummer kort brugerens holdning inden du går videre til næste emne
- Du behøver ikke dække alle 24 emner — 5-8 udvekslinger er nok
- Når du har nok information, skriv præcis: [KLAR TIL MATCH]
- Svar ALTID på dansk
- Start med at byde velkommen og stille dit første spørgsmål`;
}
```

**Step 2: Integrate system prompt into the adapter**

Update the `ChatModelAdapter` to prepend the system prompt to the message history. The system prompt is loaded from `questions.json` at startup.

**Step 3: Ensure `public/questions.json` exists**

```bash
cp data/questions.json public/questions.json
```

**Step 4: Commit**

```bash
git add src/lib/system-prompt.ts public/questions.json
git commit -m "feat: add system prompt for Danish political conversation flow"
```

---

## Task 11: Candidate matching and results display

**Files:**
- Create: `src/components/CandidateResults.tsx`
- Modify: `src/App.tsx`

**Step 1: Write the CandidateResults component**

When the LLM signals `[KLAR TIL MATCH]`, extract the user's conversation, embed it, and display matches.

```tsx
// src/components/CandidateResults.tsx
import type { CandidateMatch } from "../lib/embeddings";

interface CandidateData {
  id: number;
  name: string;
  party: string;
  partyLetter: string;
  area: string;
  age?: number;
  occupation?: string;
  pitch?: string;
}

interface Props {
  matches: CandidateMatch[];
  candidates: Map<number, CandidateData>;
}

const PARTY_COLORS: Record<string, string> = {
  a: "#A82721", // Socialdemokratiet
  b: "#733280", // Radikale Venstre
  c: "#00583C", // Konservative
  d: "#00505C", // Nye Borgerlige
  f: "#E4007F", // SF
  i: "#00B0CA", // Liberal Alliance
  k: "#F29D00", // Kristendemokraterne
  m: "#532380", // Moderaterne
  o: "#005078", // Dansk Folkeparti
  v: "#004A98", // Venstre
  ø: "#E5001A", // Enhedslisten
  å: "#2B8738", // Alternativet
};

export function CandidateResults({ matches, candidates }: Props) {
  return (
    <div className="p-4 space-y-3">
      <h2 className="text-lg font-bold text-gray-900">Dine top-matches</h2>
      {matches.map((match, i) => {
        const c = candidates.get(match.id);
        const color = PARTY_COLORS[match.party?.toLowerCase()] || "#666";
        return (
          <div
            key={match.id}
            className="flex items-start gap-3 p-3 rounded-lg border bg-white shadow-sm"
          >
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm shrink-0"
              style={{ backgroundColor: color }}
            >
              {i + 1}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="font-semibold text-gray-900">{match.name}</span>
                <span className="text-sm text-gray-500">({c?.partyLetter?.toUpperCase()})</span>
                <span className="ml-auto text-sm font-medium text-red-700">
                  {Math.round(match.score * 100)}%
                </span>
              </div>
              {c?.occupation && (
                <p className="text-sm text-gray-500">{c.occupation}</p>
              )}
              {c?.area && (
                <p className="text-xs text-gray-400">{c.area}</p>
              )}
              {c?.pitch && (
                <p className="text-sm text-gray-600 mt-1 line-clamp-2">{c.pitch}</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
```

**Step 2: Wire matching logic into App**

Add a mechanism to detect `[KLAR TIL MATCH]` in LLM output, then:
1. Concatenate all user messages from the conversation
2. Embed with EmbeddingGemma via the embedding worker
3. Run cosine similarity against pre-computed candidate embeddings
4. Display `CandidateResults` below the chat

**Step 3: Commit**

```bash
git add src/components/CandidateResults.tsx src/App.tsx
git commit -m "feat: add candidate matching and results display"
```

---

## Task 12: WebGPU detection and fallback

**Files:**
- Create: `src/lib/webgpu.ts`
- Modify: `src/App.tsx`

**Step 1: Add WebGPU check**

```typescript
// src/lib/webgpu.ts
export async function checkWebGPU(): Promise<{ supported: boolean; reason?: string }> {
  if (!navigator.gpu) {
    return { supported: false, reason: "Din browser understøtter ikke WebGPU. Brug Chrome 113+ eller Edge 113+." };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { supported: false, reason: "Ingen WebGPU-adapter fundet. Prøv at opdatere din browser." };
    }
    return { supported: true };
  } catch {
    return { supported: false, reason: "WebGPU kunne ikke initialiseres." };
  }
}
```

**Step 2: Show fallback UI in App if WebGPU is unavailable**

A centered message explaining requirements + link to Chrome.

**Step 3: Commit**

```bash
git add src/lib/webgpu.ts src/App.tsx
git commit -m "feat: add WebGPU detection with Danish fallback message"
```

---

## Task 13: Styling and polish

**Files:**
- Modify: `src/index.css`
- Modify: `index.html`

**Step 1: Style the app**

- Danish red accent (`#c8102e`) for header and match scores
- Clean thread styling via assistant-ui CSS variables
- Mobile-first responsive layout (`h-dvh` for proper mobile viewport)
- Loading overlay with file-by-file download progress bars
- Smooth transitions on progress bars

**Step 2: Set `<html lang="da">` and proper meta tags in `index.html`**

**Step 3: Commit**

```bash
git add src/index.css index.html
git commit -m "feat: add responsive styling with Danish color theme"
```

---

## Task 14: End-to-end smoke test

**Step 1: Start dev server**

```bash
npm run dev
```

Open http://localhost:5173 in Chrome (WebGPU required).

**Step 2: Verify:**
- [ ] WebGPU check runs — shows fallback on unsupported browsers
- [ ] Loading overlay shows with per-file progress bars
- [ ] Both models download and initialize (LLM ~400MB, Embeddings ~175MB)
- [ ] assistant-ui Thread renders with streaming token output
- [ ] LLM greets user in Danish and asks first question
- [ ] User can chat naturally about political views
- [ ] After 5-8 exchanges, LLM signals `[KLAR TIL MATCH]`
- [ ] Candidate matches appear with scores and party colors
- [ ] Works on mobile viewport (responsive layout)

**Step 3: Fix any issues found**

**Step 4: Commit fixes**

---

## Future tasks (post-MVP)

Not part of this plan, but noted for future iterations:

1. **Per-question embeddings** — more granular matching by embedding each question/answer pair separately
2. **Candidate detail cards** — photos (from TV2 CDN), full pitch text, expandable answer details
3. **Local question support** — let user select storkreds for local question matching
4. **Share results** — shareable URL with encoded match results
5. **PWA + offline** — service worker for offline use after initial model download
6. **Model caching UX** — show "cached" indicator on return visits (Cache API/IndexedDB)
7. **Deploy** — Cloudflare Pages with proper caching headers for large model files
8. **WASM fallback** — fall back to WASM backend for browsers without WebGPU (slower but universal)
9. **Streaming match explanation** — after matches are found, have the LLM explain why the top candidates match
