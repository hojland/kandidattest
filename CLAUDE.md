# Kandidattest

## What is this?

A fully client-side Danish election candidate matching tool. Instead of the traditional 1-5 scale questionnaire, users chat with a small LLM that runs entirely in-browser via WebGPU. The LLM asks about political views, then matches users to candidates using pre-computed embeddings.

## Tech stack

- **Vite + vanilla TypeScript** — no framework
- **Transformers.js v3** with WebGPU backend
- **Chat LLM:** `onnx-community/Qwen3-0.6B` (quantized, ~400MB, runs in-browser)
- **Embedding model:** `Xenova/multilingual-e5-small` (~130MB, supports Danish)
- **Matching:** Cosine similarity on pre-computed candidate profile embeddings

## Architecture

Zero backend. Everything runs client-side:
1. LLM conducts conversation about political views in Danish
2. User responses are embedded with e5-small
3. Compared against pre-computed candidate embeddings
4. Top matches displayed with scores

## Data

- 918 candidates from TV2's FV2026 kandidattest
- 24 national questions with -2 to +2 answer scale
- Free-text comments from 689 candidates
- Raw data in `data/candidates_raw.json` (gitignored, 6MB)
- Processed data in `public/` directory

## Commands

```bash
npm run dev       # Start dev server
npm run build     # Build for production
```

## Key directories

- `src/` — TypeScript source
- `public/` — Static assets (candidate data, embeddings, questions)
- `data/` — Raw/processed data (not all committed)
- `scripts/` — Python data processing scripts
- `docs/plans/` — Implementation plans
