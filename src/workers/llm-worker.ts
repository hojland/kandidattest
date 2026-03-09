import { pipeline, TextStreamer } from "@huggingface/transformers";

type Message = { role: string; content: string };

class LLMPipeline {
  static instance: Awaited<ReturnType<typeof pipeline>> | null = null;

  static async getInstance(progressCallback?: (data: unknown) => void) {
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
