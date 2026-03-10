export interface CandidateMatch {
  id: number;
  name: string;
  party: string;
  score: number;
}

interface EmbeddingIndex {
  dim: number;
  count: number;
  candidates: Array<{ id: number; name: string; party: string; area: string }>;
}

type ProgressCallback = (data: {
  status?: string;
  file?: string;
  progress?: number;
}) => void;

export class EmbeddingManager {
  private worker: Worker;
  private candidateEmbeddings: Float32Array | null = null;
  private index: EmbeddingIndex | null = null;
  private onProgress: ProgressCallback | undefined;
  /** Mean embedding across all candidates, used for centering */
  private meanEmbedding: Float32Array | null = null;

  constructor(onProgress?: ProgressCallback) {
    this.onProgress = onProgress;
    this.worker = new Worker(
      new URL("../workers/embedding-worker.ts", import.meta.url),
      { type: "module" },
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

    // Center candidate embeddings to remove shared signal
    // (same language, same domain, same format all compress similarity into a tiny band)
    this.centerEmbeddings();

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

  /**
   * Compute the mean embedding across all candidates, subtract it from
   * every candidate vector, then re-normalize to unit length.
   *
   * This removes the "average candidate" signal (shared language, domain,
   * format) so that cosine similarity reflects actual political differences
   * rather than surface-level text similarity.
   */
  private centerEmbeddings(): void {
    const emb = this.candidateEmbeddings!;
    const { dim, count } = this.index!;

    // 1. Compute mean vector
    const mean = new Float32Array(dim);
    for (let i = 0; i < count; i++) {
      const offset = i * dim;
      for (let d = 0; d < dim; d++) {
        mean[d] += emb[offset + d];
      }
    }
    for (let d = 0; d < dim; d++) {
      mean[d] /= count;
    }
    this.meanEmbedding = mean;

    // 2. Subtract mean and re-normalize each candidate
    for (let i = 0; i < count; i++) {
      const offset = i * dim;
      // Subtract mean
      for (let d = 0; d < dim; d++) {
        emb[offset + d] -= mean[d];
      }
      // L2 normalize
      let norm = 0;
      for (let d = 0; d < dim; d++) {
        norm += emb[offset + d] * emb[offset + d];
      }
      norm = Math.sqrt(norm);
      if (norm > 0) {
        for (let d = 0; d < dim; d++) {
          emb[offset + d] /= norm;
        }
      }
    }

    console.log("[EMB] Centered %d candidate embeddings (dim=%d)", count, dim);
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

  findMatches(
    userEmbedding: number[],
    topK = 10,
    filterArea?: string,
  ): CandidateMatch[] {
    if (!this.candidateEmbeddings || !this.index || !this.meanEmbedding) return [];
    const { dim, count, candidates } = this.index;
    const mean = this.meanEmbedding;

    // Center the user embedding using the same mean, then normalize
    const centered = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      centered[d] = userEmbedding[d] - mean[d];
    }
    let norm = 0;
    for (let d = 0; d < dim; d++) {
      norm += centered[d] * centered[d];
    }
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let d = 0; d < dim; d++) {
        centered[d] /= norm;
      }
    }

    // Compute raw cosine similarity (dot product of unit vectors)
    const rawScores: number[] = [];
    const results: { idx: number; score: number }[] = [];

    for (let i = 0; i < count; i++) {
      if (filterArea && candidates[i].area !== filterArea) continue;

      let dot = 0;
      for (let d = 0; d < dim; d++) {
        dot += centered[d] * this.candidateEmbeddings[i * dim + d];
      }
      rawScores.push(dot);
      results.push({ idx: i, score: dot });
    }

    // Percentile-based rescaling: map observed range to [0, 1]
    // Use p2/p98 to clip outliers and give a robust spread
    rawScores.sort((a, b) => a - b);
    const n = rawScores.length;
    const p2 = rawScores[Math.floor(n * 0.02)];
    const p98 = rawScores[Math.floor(n * 0.98)];
    const range = p98 - p2;

    const matches: CandidateMatch[] = results.map(({ idx, score }) => {
      const scaled = range > 0
        ? Math.max(0, Math.min(1, (score - p2) / range))
        : 0.5;
      return { ...candidates[idx], score: scaled };
    });

    matches.sort((a, b) => b.score - a.score);
    return topK > 0 ? matches.slice(0, topK) : matches;
  }
}
