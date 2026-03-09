import { AutoModel, AutoTokenizer, Tensor } from "@huggingface/transformers";

let model: InstanceType<typeof AutoModel> | null = null;
let tokenizer: InstanceType<typeof AutoTokenizer> | null = null;

self.addEventListener("message", async (event: MessageEvent) => {
  const { type, text, id } = event.data;

  if (type === "load") {
    tokenizer = await AutoTokenizer.from_pretrained("onnx-community/embeddinggemma-300m-ONNX");
    model = await AutoModel.from_pretrained("onnx-community/embeddinggemma-300m-ONNX", {
      dtype: "q4f16",
      device: "wasm",
      progress_callback: (data: unknown) => {
        self.postMessage({ type: "progress", ...(data as Record<string, unknown>) });
      },
    });
    self.postMessage({ type: "ready" });
    return;
  }

  if (type === "embed" && model && tokenizer) {
    const inputs = await tokenizer(text, { padding: true, truncation: true });
    const output = await model(inputs);
    const lastHiddenState = output.last_hidden_state as Tensor;

    // Mean pooling: average over sequence length (dim 1), get [1, 768]
    // Then L2 normalize
    const data = lastHiddenState.data as Float32Array;
    const [, seqLen, dim] = lastHiddenState.dims;
    const embedding = new Float32Array(dim);

    for (let d = 0; d < dim; d++) {
      let sum = 0;
      for (let s = 0; s < seqLen; s++) {
        sum += data[s * dim + d];
      }
      embedding[d] = sum / seqLen;
    }

    // L2 normalize (guard against zero-norm)
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += embedding[d] * embedding[d];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let d = 0; d < dim; d++) embedding[d] /= norm;
    }

    self.postMessage({ type: "embedding", embedding: Array.from(embedding), id });
  }
});
