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

  findMatches(
    userEmbedding: number[],
    topK = 10,
    filterArea?: string,
  ): CandidateMatch[] {
    if (!this.candidateEmbeddings || !this.index) return [];
    const { dim, count, candidates } = this.index;
    const results: CandidateMatch[] = [];

    for (let i = 0; i < count; i++) {
      if (filterArea && candidates[i].area !== filterArea) continue;

      let dot = 0;
      for (let d = 0; d < dim; d++) {
        dot += userEmbedding[d] * this.candidateEmbeddings[i * dim + d];
      }
      results.push({ ...candidates[i], score: dot });
    }

    return results.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}
