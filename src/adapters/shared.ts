/**
 * Shared utilities for both local and API adapters.
 */

/** Strip <think>...</think> blocks from streamed text, returning thinking content. */
export function stripThinking(raw: string): { text: string; isThinking: boolean; thinkingText: string } {
  let thinkingText = "";

  // Extract completed thinking blocks
  const completedThinks = raw.match(/<think>([\s\S]*?)<\/think>/g);
  if (completedThinks) {
    thinkingText = completedThinks.map(m => m.replace(/<\/?think>/g, "")).join("\n");
  }

  let text = raw.replace(/<think>[\s\S]*?<\/think>/g, "");
  const isThinking = /<think>(?![\s\S]*<\/think>)/.test(raw);
  if (isThinking) {
    const partialMatch = raw.match(/<think>([\s\S]*)$/);
    if (partialMatch) {
      thinkingText += (thinkingText ? "\n" : "") + partialMatch[1];
    }
    text = text.replace(/<think>[\s\S]*$/, "");
  }

  return { text: text.trim(), isThinking, thinkingText: thinkingText.trim() };
}

/** Thinking text marker prefix for the UI. */
export const THINKING_PREFIX = "💭\n";

/** Parse [VALG: opt1 | opt2 | opt3] markup from LLM output. */
export function parseChoices(text: string): {
  textBefore: string;
  options: string[];
  hasCompleteChoice: boolean;
  hasPartialChoice: boolean;
} {
  const completeMatch = text.match(/^([\s\S]*?)\[VALG:\s*(.+?)\]\s*$/);
  if (completeMatch) {
    const textBefore = completeMatch[1].trim();
    const options = completeMatch[2].split("|").map((o) => o.trim()).filter(Boolean);
    return { textBefore, options, hasCompleteChoice: true, hasPartialChoice: false };
  }

  const partialMatch = text.match(/^([\s\S]*?)\[VALG:[^\]]*$/);
  if (partialMatch) {
    return { textBefore: partialMatch[1].trim(), options: [], hasCompleteChoice: false, hasPartialChoice: true };
  }

  return { textBefore: text, options: [], hasCompleteChoice: false, hasPartialChoice: false };
}

let choiceCounter = 0;
export function nextChoiceId(): string {
  return `choice_${++choiceCounter}`;
}
