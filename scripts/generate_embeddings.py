#!/usr/bin/env python3
"""Generate embeddings for candidate profile texts using EmbeddingGemma-300M.

Uses the full-precision model server-side. The browser will use the q4f16
quantized ONNX version for user text embedding. Cosine similarity between
full and quantized embeddings is high enough for ranking.

Requires HF_TOKEN env var (model is gated). Set in .env file.
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "google/embeddinggemma-300m"
HF_TOKEN = os.environ.get("HF_TOKEN")
EMBEDDING_DIM = 768  # Full dim; could truncate to 256 via MRL if needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32, token=HF_TOKEN)
model.eval()

with open("data/candidates.json") as f:
    candidates = json.load(f)

texts = [c["profileText"] for c in candidates]
index_entries = [{"id": c["id"], "name": c["name"], "party": c["party"], "area": c["area"]} for c in candidates]

print(f"Generating embeddings for {len(texts)} candidates...")

BATCH_SIZE = 32
all_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
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
    json.dump(
        {
            "dim": EMBEDDING_DIM,
            "count": len(index_entries),
            "candidates": index_entries,
        },
        f,
        ensure_ascii=False,
    )

size_mb = embeddings_np.nbytes / 1024 / 1024
print(f"Saved {embeddings_np.shape} embeddings ({size_mb:.1f} MB)")
# Expected: ~824 × 768 = ~2.4 MB
