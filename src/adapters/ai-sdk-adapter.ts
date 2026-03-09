import type { ChatModelAdapter } from "@assistant-ui/react";
import { streamText, type LanguageModel } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { Provider } from "../lib/provider";
import { stripThinking, parseChoices, nextChoiceId, THINKING_PREFIX } from "./shared";

function createModel(provider: Provider): LanguageModel {
  switch (provider.kind) {
    case "anthropic":
      return createAnthropic({
        apiKey: provider.apiKey,
      })(provider.model);

    case "google":
      return createGoogleGenerativeAI({
        apiKey: provider.apiKey,
      })(provider.model);

    case "openai":
      return createOpenAI({
        apiKey: provider.apiKey,
      })(provider.model);

    case "openai-compatible":
      return createOpenAI({
        apiKey: provider.apiKey,
        baseURL: provider.baseUrl,
      })(provider.model);
  }
}

export function createAISDKAdapter(
  provider: Provider,
  systemPrompt: string,
): ChatModelAdapter {
  const model = createModel(provider);

  return {
    async *run({ messages, abortSignal }) {
      const chatMessages = messages.map((m) => ({
        role: m.role as "user" | "assistant",
        content: m.content
          .filter((p): p is { type: "text"; text: string } => p.type === "text")
          .map((p) => p.text)
          .join(""),
      }));

      const result = streamText({
        model,
        system: systemPrompt || undefined,
        messages: chatMessages,
        abortSignal,
        temperature: 0.7,
        maxOutputTokens: 4096,
        providerOptions: {
          anthropic: {
            thinking: { type: "enabled", budgetTokens: 2048 },
          },
          google: {
            thinkingConfig: { thinkingBudget: 2048 },
          },
        },
      });

      let fullText = "";
      let reasoningText = "";

      for await (const part of result.fullStream) {
        if (part.type === "reasoning-start" || part.type === "reasoning-end") {
          continue;
        } else if (part.type === "reasoning-delta") {
          reasoningText += part.text;
        } else if (part.type === "text-delta") {
          fullText += part.text;
        } else {
          continue;
        }

        // Process accumulated text
        const { text, thinkingText } = stripThinking(fullText);
        const allThinking = reasoningText + (thinkingText ? "\n" + thinkingText : "");

        const buildParts = (mainText: string): Array<{ type: "text"; text: string }> => {
          const parts: Array<{ type: "text"; text: string }> = [];
          if (allThinking) {
            parts.push({ type: "text" as const, text: THINKING_PREFIX + allThinking });
          }
          if (mainText) {
            parts.push({ type: "text" as const, text: mainText });
          }
          return parts.length ? parts : [{ type: "text" as const, text: "" }];
        };

        // No visible content yet — show thinking progress
        if (!text) {
          if (allThinking) {
            yield { content: [{ type: "text" as const, text: THINKING_PREFIX + allThinking }] };
          }
          continue;
        }

        // Check for choice markup
        const { textBefore, options, hasCompleteChoice, hasPartialChoice } =
          parseChoices(text);

        if (hasPartialChoice) {
          const parts = buildParts(textBefore);
          if (!parts.some((p) => p.text.startsWith(THINKING_PREFIX))) {
            parts.push({ type: "text" as const, text: THINKING_PREFIX + "Forbereder valgmuligheder..." });
          }
          yield { content: parts };
        } else if (hasCompleteChoice && options.length > 0) {
          const parts: Array<unknown> = [];
          if (allThinking) parts.push({ type: "text" as const, text: THINKING_PREFIX + allThinking });
          if (textBefore) parts.push({ type: "text" as const, text: textBefore });
          parts.push({
            type: "tool-call" as const,
            toolCallId: nextChoiceId(),
            toolName: "user_choice",
            args: { options },
            argsText: JSON.stringify({ options }),
          });
          yield {
            content: parts as Array<{ type: "text"; text: string }>,
            status: { type: "requires-action" as const, reason: "tool-calls" as const },
          };
        } else {
          yield { content: buildParts(text) };
        }
      }

      // If stream ended with only reasoning and no content
      if (reasoningText && !fullText.trim()) {
        yield {
          content: [{
            type: "text" as const,
            text: THINKING_PREFIX + reasoningText + "\n\n---\n\nModellen tænkte, men genererede intet svar. Prøv igen.",
          }],
        };
      }
    },
  };
}
