#!/usr/bin/env python3
"""Generate candidate embeddings using Google's text-embedding-004 API.

Uses the same candidate data as generate_embeddings.py but produces embeddings
in a different vector space. The output shares embedding_index.json (same
candidate order) but writes to a separate embeddings_google.bin file.

Requires GOOGLE_API_KEY env var.
"""

import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from google import genai

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY env var")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072
BATCH_SIZE = 100  # Google API supports up to 100 texts per request

with open("data/candidates.json") as f:
    candidates = json.load(f)

texts = [c["profileText"] for c in candidates]
index_entries = [{"id": c["id"], "name": c["name"], "party": c["party"], "area": c["area"]} for c in candidates]

print(f"Generating embeddings for {len(texts)} candidates using {MODEL}...")

all_embeddings = []
total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1

    for attempt in range(5):
        try:
            result = client.models.embed_content(
                model=MODEL,
                contents=batch,
            )
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/5)...")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError(f"Failed batch {batch_num} after 5 retries")

    batch_embeddings = [e.values for e in result.embeddings]
    all_embeddings.extend(batch_embeddings)
    print(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)")
    if i + BATCH_SIZE < len(texts):
        time.sleep(5)

embeddings_np = np.array(all_embeddings, dtype=np.float32)

# L2 normalize
norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
norms[norms == 0] = 1
embeddings_np = embeddings_np / norms

# Save as flat float32 binary (same format as embeddings.bin)
embeddings_np.tofile("public/embeddings_google.bin")

# Verify dimensions match expected
assert embeddings_np.shape == (len(candidates), EMBEDDING_DIM), (
    f"Unexpected shape: {embeddings_np.shape}, expected ({len(candidates)}, {EMBEDDING_DIM})"
)

size_mb = embeddings_np.nbytes / 1024 / 1024
print(f"Saved {embeddings_np.shape} embeddings to public/embeddings_google.bin ({size_mb:.1f} MB)")
print("Reuses existing public/embedding_index.json (same candidate order)")
