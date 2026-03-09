# Kandidattest

## What is this?

A fully client-side Danish election candidate matching tool for FV2026. Instead of the traditional 1-5 scale questionnaire, users chat with a small LLM that runs entirely in-browser via WebGPU. The LLM asks about political views in Danish, then matches users to candidates using embedding similarity.

## Tech stack

- **Vite + React + TypeScript**
- **assistant-ui** (`@assistant-ui/react`) with `LocalRuntime` — chat UI with streaming, markdown, auto-scroll
- **shadcn/ui + Tailwind CSS** — styling
- **Transformers.js v3** with WebGPU backend
- **Chat LLM:** `onnx-community/Qwen3-0.6B` (q4f16, ~400MB, runs in-browser)
- **Embedding model:** `onnx-community/embeddinggemma-300m-ONNX` (q4f16, ~175MB, MTEB Multilingual 61.15)
- **Pre-computed embeddings** generated once server-side with `google/embeddinggemma-300m`

## Architecture

Zero backend. Everything runs client-side:
1. assistant-ui `LocalRuntime` + `ChatModelAdapter` bridges to LLM web worker
2. Qwen3-0.6B runs in a web worker via Transformers.js WebGPU, streams tokens back
3. After 5-8 conversation exchanges, user text is embedded with EmbeddingGemma in a second web worker
4. Cosine similarity against pre-computed candidate embeddings → ranked match list
5. Results shown as candidate cards with party colors and match percentages

## Data

- 918 candidates scraped from TV2's FV2026 kandidattest
- 24 national questions with -2 to +2 answer scale
- Free-text comments from 689 candidates
- Raw data in `data/candidates_raw.json` (gitignored, 6MB)
- Processed data + embeddings in `public/` directory

## Commands

```bash
npm run dev       # Start dev server (requires WebGPU-capable browser)
npm run build     # Build for production

# Data processing (one-time)
python3 scripts/process_candidates.py     # Process raw candidate data
python3 scripts/generate_embeddings.py    # Generate candidate embeddings
```

## Key directories

- `src/` — React + TypeScript source
  - `src/workers/` — Web workers for LLM and embedding inference
  - `src/adapters/` — ChatModelAdapter bridging workers to assistant-ui
  - `src/components/` — React components (results, etc.)
  - `src/lib/` — Utilities (embeddings manager, system prompt, WebGPU check)
- `public/` — Static assets (candidate data, embeddings, questions)
- `data/` — Raw/processed data (not all committed)
- `scripts/` — Python data processing scripts
- `docs/plans/` — Implementation plans
