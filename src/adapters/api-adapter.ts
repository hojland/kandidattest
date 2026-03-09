import type { ChatModelAdapter } from "@assistant-ui/react";
import { stripThinking, parseChoices, nextChoiceId } from "./shared";

export function createApiAdapter(
  baseUrl: string,
  apiKey: string,
  model: string,
  systemPrompt: string,
): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const apiMessages = [
        ...(systemPrompt ? [{ role: "system" as const, content: systemPrompt }] : []),
        ...messages.map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content
            .filter((p): p is { type: "text"; text: string } => p.type === "text")
            .map((p) => p.text)
            .join(""),
        })),
      ];

      const response = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: apiMessages,
          stream: true,
          temperature: 0.7,
          max_tokens: 1024,
        }),
        signal: abortSignal,
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`API error ${response.status}: ${err}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let fullText = "";
      let buffer = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") continue;

            try {
              const chunk = JSON.parse(data);
              const delta = chunk.choices?.[0]?.delta?.content;
              if (!delta) continue;

              fullText += delta;
              const { text, isThinking } = stripThinking(fullText);

              if (isThinking && !text) {
                yield { content: [{ type: "text" as const, text: "\u{1F4AD} T\u00e6nker..." }] };
                continue;
              }

              const displayText = isThinking ? text + "\n\n\u{1F4AD} T\u00e6nker..." : text;
              const { textBefore, options, hasCompleteChoice, hasPartialChoice } =
                parseChoices(displayText);

              if (hasPartialChoice) {
                const content: Array<{ type: "text"; text: string }> = [];
                if (textBefore) content.push({ type: "text" as const, text: textBefore });
                content.push({ type: "text" as const, text: "\u{1F4AD} T\u00e6nker..." });
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
                yield { content: [{ type: "text" as const, text: displayText }] };
              }
            } catch { /* skip malformed SSE chunks */ }
          }
        }
      } finally {
        reader.releaseLock();
      }
    },
  };
}
