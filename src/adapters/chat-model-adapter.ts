import type { ChatModelAdapter } from "@assistant-ui/react";
import { stripThinking, parseChoices, nextChoiceId } from "./shared";

export function createLLMAdapter(
  worker: Worker,
  systemPrompt: string,
): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const history = [
        ...(systemPrompt
          ? [{ role: "system" as const, content: systemPrompt }]
          : []),
        ...messages.map((m) => ({
          role: m.role,
          content: m.content
            .filter(
              (p): p is { type: "text"; text: string } => p.type === "text",
            )
            .map((p) => p.text)
            .join(""),
        })),
      ];

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

          const { text, isThinking } = stripThinking(fullText);

          if (isThinking && !text) {
            yield {
              content: [{ type: "text" as const, text: "💭 Tænker..." }],
            };
            continue;
          }

          const displayText = isThinking ? text + "\n\n💭 Tænker..." : text;
          const { textBefore, options, hasCompleteChoice, hasPartialChoice } =
            parseChoices(displayText);

          if (hasPartialChoice) {
            const content: Array<{ type: "text"; text: string }> = [];
            if (textBefore) content.push({ type: "text" as const, text: textBefore });
            content.push({ type: "text" as const, text: "💭 Tænker..." });
            yield { content };
          } else if (hasCompleteChoice && options.length > 0) {
            const toolCallId = nextChoiceId();
            const content: Array<any> = [];
            if (textBefore) content.push({ type: "text" as const, text: textBefore });
            content.push({
              type: "tool-call" as const,
              toolCallId,
              toolName: "user_choice",
              args: { options },
              argsText: JSON.stringify({ options }),
            });
            yield {
              content,
              status: { type: "requires-action" as const, reason: "tool-calls" as const },
            };
          } else {
            yield {
              content: [{ type: "text" as const, text: displayText }],
            };
          }
        }
      } finally {
        worker.removeEventListener("message", onMessage);
      }
    },
  };
}
