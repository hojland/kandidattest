# Kandidattest MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fully client-side Danish election candidate matcher that uses a conversational chat interface powered by a small quantized LLM (Qwen3-0.6B via Transformers.js with WebGPU), matching users to candidates via pre-computed embeddings.

**Architecture:** Static site (Vite + vanilla TypeScript) with zero backend. A small LLM runs in-browser via Transformers.js WebGPU to conduct a natural conversation about political views. User responses are embedded client-side and compared against pre-computed candidate answer embeddings using cosine similarity. Results show ranked candidate matches with explanations.

**Tech Stack:**
- Vite + TypeScript (no framework — keep it minimal)
- Transformers.js v3 with WebGPU backend
- Chat LLM: `onnx-community/Qwen3-0.6B` (ONNX quantized, ~400MB)
- Embedding model: `Xenova/multilingual-e5-small` (~130MB, supports Danish)
- Pre-computed embeddings generated once via Python script
- Deployed as static files (GitHub Pages / Cloudflare Pages)

---

## Data we have

- `data/candidates_raw.json` — 918 candidates, 24 national questions (answer scale -2 to +2), free-text comments (689 candidates), priorities, local questions
- Question keys: `tv2-fv26-danmark-1` through `tv2-fv26-danmark-24`
- **Missing:** Question text strings (only keys exist in data). Must be scraped from the rendered TV2 page.

## Architecture overview

```
┌─────────────────────────────────────────────────┐
│  Browser (all client-side)                      │
│                                                 │
│  ┌──────────┐    ┌──────────────┐               │
│  │ Chat UI  │◄──►│ Qwen3-0.6B   │               │
│  │          │    │ (WebGPU)     │               │
│  └────┬─────┘    └──────────────┘               │
│       │                                         │
│  ┌────▼─────┐    ┌──────────────┐               │
│  │ User     │    │ Candidate    │               │
│  │ Embedding│◄──►│ Embeddings   │               │
│  │ (e5)     │    │ (precomputed)│               │
│  └──────────┘    └──────────────┘               │
│                                                 │
│  Matching: cosine similarity on 24 question     │
│  embeddings → ranked candidate list             │
└─────────────────────────────────────────────────┘
```

## Matching strategy

1. The LLM converses with the user about political topics (guided by the 24 question themes)
2. After enough signal, the system extracts the user's stance on each topic
3. User stances are embedded with e5-small
4. Cosine similarity against pre-computed candidate embeddings per question
5. Weighted average across questions → ranked match list
6. The LLM explains the top matches conversationally

**Simpler alternative (Phase 1):** Skip per-question embedding. Instead:
- Candidate "profile text" = concatenation of answers + comments + pitch
- User "profile text" = concatenation of chat responses about political views
- Single embedding comparison for overall match
- This is simpler and gets us to MVP faster

---

## Task 1: Project scaffolding

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `vite.config.ts`
- Create: `index.html`
- Create: `src/main.ts`
- Create: `src/style.css`
- Create: `.gitignore`

**Step 1: Initialize project with Vite**

```bash
cd /Users/hojland/kandidattest
npm create vite@latest . -- --template vanilla-ts
```

Accept overwrite prompts for existing files.

**Step 2: Install dependencies**

```bash
npm install
npm install @huggingface/transformers
```

**Step 3: Configure Vite for WebGPU workers**

Edit `vite.config.ts`:
```typescript
import { defineConfig } from "vite";

export default defineConfig({
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },
  worker: {
    format: "es",
  },
});
```

**Step 4: Update .gitignore**

Ensure `node_modules/`, `dist/`, and `data/candidates_raw.json` are gitignored. The raw data is 6MB — only processed/compressed outputs go in the repo.

```
node_modules/
dist/
data/candidates_raw.json
```

**Step 5: Verify dev server starts**

```bash
npm run dev
```

Expected: Vite dev server starts on localhost:5173

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: scaffold Vite + TypeScript project with Transformers.js"
```

---

## Task 2: Extract question texts from TV2

**Files:**
- Create: `scripts/scrape_questions.py`
- Create: `data/questions.json`

**Step 1: Write scraper script**

Uses agent-browser to navigate to a candidate page and extract rendered question texts from the DOM. The questions are rendered in the HTML but not in the data-props JSON.

```python
#!/usr/bin/env python3
"""Extract question texts from TV2's kandidattest rendered page."""
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

# Open a candidate page that has all national answers
ab("open", "https://nyheder.tv2.dk/folketingsvalg/kandidat/109366")
time.sleep(5)

# Extract question texts from rendered DOM
# The page renders questions with their text labels
result = ab("eval", """
(() => {
    // Find all question elements on the candidate page
    // Questions are rendered as text near the answer indicators
    const questions = {};
    // Look for elements containing question text
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

This is a best-effort script — the exact DOM selectors will need to be discovered during implementation by inspecting the rendered page with `agent-browser snapshot`.

**Step 2: Manually verify and save questions**

The script output needs manual review. Save the final mapping as:

```json
{
  "tv2-fv26-danmark-1": "Danmark bør bruge flere penge på forsvar",
  "tv2-fv26-danmark-2": "...",
  ...
}
```

Save to `data/questions.json`.

**Step 3: Commit**

```bash
git add data/questions.json scripts/scrape_questions.py
git commit -m "feat: extract question texts from TV2 kandidattest"
```

---

## Task 3: Process candidate data

**Files:**
- Create: `scripts/process_candidates.py`
- Create: `data/candidates.json` (processed, compact)

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
            # Build profile text for embedding
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

with open("data/candidates.json", "w") as f:
    json.dump(candidates, f, ensure_ascii=False)

# Also save a version without profileText for the frontend (smaller)
for c in candidates:
    del c["profileText"]

with open("public/candidates.json", "w") as f:
    json.dump(candidates, f, ensure_ascii=False)

print(f"Saved public/candidates.json ({len(candidates)} candidates)")
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

```python
#!/usr/bin/env python3
"""Generate embeddings for candidate profile texts using sentence-transformers."""
import json
import numpy as np

# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")

with open("data/candidates.json") as f:
    candidates = json.load(f)

texts = []
index = []
for c in candidates:
    text = f"passage: {c['profileText']}"  # e5 prefix format
    texts.append(text)
    index.append({"id": c["id"], "name": c["name"], "party": c["party"]})

print(f"Generating embeddings for {len(texts)} candidates...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

# Save as float32 binary (compact)
embeddings_f32 = embeddings.astype(np.float32)
embeddings_f32.tofile("public/embeddings.bin")

# Save index mapping
with open("public/embedding_index.json", "w") as f:
    json.dump({
        "dim": embeddings.shape[1],  # 384 for e5-small
        "count": len(index),
        "candidates": index,
    }, f, ensure_ascii=False)

size_mb = embeddings_f32.nbytes / 1024 / 1024
print(f"Saved {embeddings.shape} embeddings ({size_mb:.1f} MB)")
```

**Step 2: Install dependencies and run**

```bash
pip install sentence-transformers numpy
python3 scripts/generate_embeddings.py
```

Expected output: ~824 embeddings × 384 dims = ~1.2 MB binary file.

**Step 3: Commit**

```bash
git add scripts/generate_embeddings.py public/embeddings.bin public/embedding_index.json
git commit -m "feat: generate candidate profile embeddings with e5-small"
```

---

## Task 5: Build chat UI

**Files:**
- Modify: `index.html`
- Modify: `src/style.css`
- Create: `src/ui.ts`

**Step 1: Build the HTML shell**

Minimal chat interface: header, message list, input area, and a loading overlay for model download.

```html
<!DOCTYPE html>
<html lang="da">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Kandidattest — Find din kandidat med AI</title>
  <link rel="stylesheet" href="/src/style.css" />
</head>
<body>
  <div id="app">
    <header>
      <h1>Kandidattest</h1>
      <p>Chat med AI om politik — find din kandidat</p>
    </header>

    <div id="loading" class="loading">
      <div class="loading-content">
        <p>Henter AI-model...</p>
        <progress id="progress" value="0" max="100"></progress>
        <p id="progress-text">0%</p>
      </div>
    </div>

    <div id="chat" class="chat hidden">
      <div id="messages" class="messages"></div>
      <form id="input-form" class="input-area">
        <input type="text" id="user-input" placeholder="Skriv dit svar..." autocomplete="off" />
        <button type="submit">Send</button>
      </form>
    </div>

    <div id="results" class="results hidden"></div>
  </div>

  <script type="module" src="/src/main.ts"></script>
</body>
</html>
```

**Step 2: Style it**

Clean, mobile-first chat UI. Danish-flag-inspired color accent. See `src/style.css`.

**Step 3: Write UI module**

`src/ui.ts` — handles DOM manipulation: adding messages, showing/hiding loading, rendering results.

```typescript
export function addMessage(role: "user" | "assistant", text: string) {
  const messages = document.getElementById("messages")!;
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

export function showLoading(show: boolean) {
  document.getElementById("loading")!.classList.toggle("hidden", !show);
  document.getElementById("chat")!.classList.toggle("hidden", show);
}

export function updateProgress(pct: number, text?: string) {
  const bar = document.getElementById("progress") as HTMLProgressElement;
  const label = document.getElementById("progress-text")!;
  bar.value = pct;
  label.textContent = text || `${Math.round(pct)}%`;
}

export function showResults(matches: Array<{name: string; party: string; score: number; area: string}>) {
  const el = document.getElementById("results")!;
  el.classList.remove("hidden");
  el.innerHTML = `
    <h2>Dine top-matches</h2>
    <ol>
      ${matches.map((m, i) => `
        <li>
          <strong>${m.name}</strong> (${m.party})
          <span class="score">${Math.round(m.score * 100)}% match</span>
          <span class="area">${m.area}</span>
        </li>
      `).join("")}
    </ol>
  `;
}
```

**Step 4: Commit**

```bash
git add index.html src/style.css src/ui.ts
git commit -m "feat: build chat UI with loading and results display"
```

---

## Task 6: Embedding worker (client-side)

**Files:**
- Create: `src/embedding-worker.ts`
- Create: `src/embeddings.ts`

**Step 1: Write the web worker for embedding inference**

The embedding model runs in a web worker to avoid blocking the UI thread.

```typescript
// src/embedding-worker.ts
import { pipeline, env } from "@huggingface/transformers";

env.allowLocalModels = false;

let embedder: Awaited<ReturnType<typeof pipeline>> | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { type, text, id } = e.data;

  if (type === "load") {
    embedder = await pipeline("feature-extraction", "Xenova/multilingual-e5-small", {
      dtype: "fp32",
      device: "webgpu",
      progress_callback: (progress: any) => {
        self.postMessage({ type: "progress", ...progress });
      },
    });
    self.postMessage({ type: "ready" });
    return;
  }

  if (type === "embed" && embedder) {
    const output = await embedder(`query: ${text}`, {
      pooling: "mean",
      normalize: true,
    });
    const embedding = Array.from(output.data as Float32Array);
    self.postMessage({ type: "embedding", embedding, id });
  }
};
```

**Step 2: Write the embeddings manager**

```typescript
// src/embeddings.ts
export class EmbeddingManager {
  private worker: Worker;
  private candidateEmbeddings: Float32Array | null = null;
  private index: { dim: number; count: number; candidates: Array<{id: number; name: string; party: string}> } | null = null;
  private pendingCallbacks = new Map<string, (emb: number[]) => void>();

  constructor(onProgress: (pct: number) => void) {
    this.worker = new Worker(new URL("./embedding-worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e) => {
      if (e.data.type === "progress" && e.data.progress) {
        onProgress(e.data.progress);
      }
      if (e.data.type === "embedding") {
        const cb = this.pendingCallbacks.get(e.data.id);
        if (cb) {
          cb(e.data.embedding);
          this.pendingCallbacks.delete(e.data.id);
        }
      }
    };
  }

  async load(): Promise<void> {
    // Load pre-computed candidate embeddings
    const [embBuf, indexData] = await Promise.all([
      fetch("/embeddings.bin").then(r => r.arrayBuffer()),
      fetch("/embedding_index.json").then(r => r.json()),
    ]);
    this.candidateEmbeddings = new Float32Array(embBuf);
    this.index = indexData;

    // Load embedding model in worker
    return new Promise((resolve) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === "ready") resolve();
      };
      this.worker.postMessage({ type: "load" });
    });
  }

  async embed(text: string): Promise<number[]> {
    const id = crypto.randomUUID();
    return new Promise((resolve) => {
      this.pendingCallbacks.set(id, resolve);
      this.worker.postMessage({ type: "embed", text, id });
    });
  }

  findMatches(userEmbedding: number[], topK = 10): Array<{id: number; name: string; party: string; score: number}> {
    if (!this.candidateEmbeddings || !this.index) return [];
    const dim = this.index.dim;
    const results: Array<{id: number; name: string; party: string; score: number}> = [];

    for (let i = 0; i < this.index.count; i++) {
      let dot = 0;
      for (let d = 0; d < dim; d++) {
        dot += userEmbedding[d] * this.candidateEmbeddings[i * dim + d];
      }
      results.push({ ...this.index.candidates[i], score: dot });
    }

    return results.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}
```

**Step 3: Commit**

```bash
git add src/embedding-worker.ts src/embeddings.ts
git commit -m "feat: add embedding worker and similarity matching"
```

---

## Task 7: LLM chat worker

**Files:**
- Create: `src/llm-worker.ts`
- Create: `src/chat.ts`

**Step 1: Write the LLM web worker**

```typescript
// src/llm-worker.ts
import { pipeline, env, TextGenerationOutput } from "@huggingface/transformers";

env.allowLocalModels = false;

let generator: Awaited<ReturnType<typeof pipeline>> | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { type, messages, id } = e.data;

  if (type === "load") {
    generator = await pipeline("text-generation", "onnx-community/Qwen3-0.6B", {
      dtype: "q4f16",
      device: "webgpu",
      progress_callback: (progress: any) => {
        self.postMessage({ type: "progress", ...progress });
      },
    });
    self.postMessage({ type: "ready" });
    return;
  }

  if (type === "generate" && generator) {
    const output = await generator(messages, {
      max_new_tokens: 256,
      temperature: 0.7,
      do_sample: true,
    }) as TextGenerationOutput;

    const lastMessage = output[0].generated_text.at(-1);
    self.postMessage({
      type: "response",
      text: lastMessage?.content || "",
      id,
    });
  }
};
```

**Step 2: Write the chat manager**

```typescript
// src/chat.ts
export class ChatManager {
  private worker: Worker;
  private history: Array<{role: string; content: string}> = [];
  private pendingCallbacks = new Map<string, (text: string) => void>();
  private systemPrompt: string;

  constructor(
    questions: Record<string, string>,
    onProgress: (pct: number) => void
  ) {
    this.worker = new Worker(new URL("./llm-worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e) => {
      if (e.data.type === "progress" && e.data.progress) {
        onProgress(e.data.progress);
      }
      if (e.data.type === "response") {
        const cb = this.pendingCallbacks.get(e.data.id);
        if (cb) {
          cb(e.data.text);
          this.pendingCallbacks.delete(e.data.id);
        }
      }
    };

    const questionList = Object.values(questions).map((q, i) => `${i + 1}. ${q}`).join("\n");
    this.systemPrompt = `Du er en venlig politisk rådgiver der hjælper danske vælgere med at finde deres kandidat til folketingsvalget 2026.

Du skal føre en naturlig samtale på dansk om politiske emner. Stil spørgsmål om brugerens holdninger til disse emner:
${questionList}

Regler:
- Stil ét spørgsmål ad gangen
- Brug et venligt, uformelt sprog
- Opsummer kort brugerens holdning før du går videre
- Efter 5-8 udvekslinger, sig at du har nok information til at finde matches
- Svar ALTID på dansk`;
  }

  async load(): Promise<void> {
    return new Promise((resolve) => {
      const origHandler = this.worker.onmessage;
      this.worker.onmessage = (e) => {
        if (e.data.type === "ready") {
          this.worker.onmessage = origHandler;
          resolve();
        }
        if (origHandler) origHandler.call(this.worker, e);
      };
      this.worker.postMessage({ type: "load" });
    });
  }

  async send(userMessage: string): Promise<string> {
    this.history.push({ role: "user", content: userMessage });

    const messages = [
      { role: "system", content: this.systemPrompt },
      ...this.history,
    ];

    const id = crypto.randomUUID();
    return new Promise((resolve) => {
      this.pendingCallbacks.set(id, (text) => {
        this.history.push({ role: "assistant", content: text });
        resolve(text);
      });
      this.worker.postMessage({ type: "generate", messages, id });
    });
  }

  getConversationSummary(): string {
    return this.history
      .filter(m => m.role === "user")
      .map(m => m.content)
      .join("\n");
  }
}
```

**Step 3: Commit**

```bash
git add src/llm-worker.ts src/chat.ts
git commit -m "feat: add LLM chat worker with Qwen3-0.6B WebGPU"
```

---

## Task 8: Wire everything together in main.ts

**Files:**
- Modify: `src/main.ts`

**Step 1: Write the main orchestration**

```typescript
// src/main.ts
import { ChatManager } from "./chat";
import { EmbeddingManager } from "./embeddings";
import { addMessage, showLoading, updateProgress, showResults } from "./ui";

async function main() {
  // Load questions
  const questions: Record<string, string> = await fetch("/questions.json").then(r => r.json());

  // Load candidate data for display
  const candidates: Array<{id: number; name: string; party: string; area: string}> =
    await fetch("/candidates.json").then(r => r.json());
  const candidateMap = new Map(candidates.map(c => [c.id, c]));

  // Initialize models
  showLoading(true);
  updateProgress(0, "Henter embedding-model...");

  const embeddings = new EmbeddingManager((pct) => updateProgress(pct * 0.5, "Embedding-model..."));
  const chat = new ChatManager(questions, (pct) => updateProgress(50 + pct * 0.5, "Chat-model..."));

  await embeddings.load();
  updateProgress(50, "Henter chat-model...");
  await chat.load();

  showLoading(false);

  // Start conversation
  const greeting = await chat.send("Hej! Jeg vil gerne finde ud af, hvilke kandidater der passer bedst til mine politiske holdninger.");
  addMessage("assistant", greeting);

  let exchangeCount = 0;

  // Handle user input
  const form = document.getElementById("input-form") as HTMLFormElement;
  const input = document.getElementById("user-input") as HTMLInputElement;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    input.disabled = true;
    addMessage("user", text);
    exchangeCount++;

    if (exchangeCount >= 8) {
      // Generate matches
      addMessage("assistant", "Tak for dine svar! Lad mig finde de kandidater der passer bedst til dig...");

      const summary = chat.getConversationSummary();
      const userEmb = await embeddings.embed(summary);
      const matches = embeddings.findMatches(userEmb, 10);

      const enrichedMatches = matches.map(m => ({
        ...m,
        area: candidateMap.get(m.id)?.area || "",
      }));

      showResults(enrichedMatches);
    } else {
      const response = await chat.send(text);
      addMessage("assistant", response);
    }

    input.disabled = false;
    input.focus();
  });
}

main().catch(console.error);
```

**Step 2: Commit**

```bash
git add src/main.ts
git commit -m "feat: wire chat, embeddings, and UI together"
```

---

## Task 9: Copy data files to public/

**Files:**
- Create: `public/questions.json` (copy from data/)
- Verify: `public/candidates.json`, `public/embeddings.bin`, `public/embedding_index.json` exist

**Step 1: Ensure all data files are in public/**

```bash
cp data/questions.json public/questions.json
ls -la public/
```

**Step 2: Commit**

```bash
git add public/
git commit -m "feat: add data files to public directory"
```

---

## Task 10: Style and polish

**Files:**
- Modify: `src/style.css`

**Step 1: Write complete styles**

Mobile-first, clean chat UI. Danish red accent (#c8102e). Dark message bubbles for assistant, light for user. Responsive layout.

Key design:
- Max-width 640px centered container
- Chat bubbles with rounded corners
- Smooth scroll for message area
- Loading overlay with progress bar
- Results as a clean ranked list with match percentages
- Works well on mobile (the primary use case for election tools)

**Step 2: Commit**

```bash
git add src/style.css
git commit -m "feat: add responsive chat UI styling"
```

---

## Task 11: End-to-end smoke test

**Step 1: Start dev server and test manually**

```bash
npm run dev
```

Open http://localhost:5173 in Chrome (WebGPU required).

**Step 2: Verify:**
- [ ] Loading overlay shows with progress
- [ ] Models download and initialize
- [ ] Chat greeting appears in Danish
- [ ] User can type and receive responses
- [ ] After ~8 exchanges, results appear
- [ ] Results show ranked candidates with match scores

**Step 3: Fix any issues found**

**Step 4: Commit any fixes**

---

## Task 12: WebGPU fallback

**Files:**
- Modify: `src/main.ts`

**Step 1: Add WebGPU detection**

```typescript
async function checkWebGPU(): Promise<boolean> {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}
```

If WebGPU is not available, show a message explaining browser requirements (Chrome 113+, Edge 113+, or Chrome Canary with flags on other platforms). Optionally fall back to WASM backend (slower but works everywhere).

**Step 2: Commit**

```bash
git add src/main.ts
git commit -m "feat: add WebGPU detection with fallback message"
```

---

## Future tasks (post-MVP)

These are NOT part of the MVP but noted for future iterations:

1. **Streaming LLM output** — show tokens as they generate instead of waiting for full response
2. **Per-question embeddings** — more granular matching instead of single profile embedding
3. **Candidate cards** — rich display with photos, party logos, and storkreds info
4. **Local question support** — let user select their storkreds for local question matching
5. **Share results** — generate a shareable URL with encoded results
6. **PWA support** — service worker for offline use after initial model download
7. **Model caching** — cache downloaded models in IndexedDB/Cache API for instant reload
8. **Deploy** — GitHub Pages or Cloudflare Pages with proper caching headers for large model files
