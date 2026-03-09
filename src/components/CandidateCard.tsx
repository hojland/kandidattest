import { useState, useEffect, useRef } from "react";
import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { CandidateData } from "../App";
import type { Provider } from "../lib/provider";

const SCORE_LABELS: Record<number, string> = {
  "-2": "Helt uenig",
  "-1": "Delvist uenig",
  "0": "Neutral",
  "1": "Delvist enig",
  "2": "Helt enig",
};

function createModel(provider: Provider) {
  switch (provider.kind) {
    case "anthropic":
      return createAnthropic({ apiKey: provider.apiKey })(provider.model);
    case "google":
      return createGoogleGenerativeAI({ apiKey: provider.apiKey })(provider.model);
    case "openai":
      return createOpenAI({ apiKey: provider.apiKey })(provider.model);
    case "openai-compatible":
      return createOpenAI({ apiKey: provider.apiKey, baseURL: provider.baseUrl })(provider.model);
  }
}

interface Props {
  candidate: CandidateData;
  provider: Provider;
  questions: Record<string, string>;
  getUserMessages: () => string[];
  comparisonCache: React.RefObject<Map<number, string>>;
}

export function CandidateCard({
  candidate,
  provider,
  questions,
  getUserMessages,
  comparisonCache,
}: Props) {
  const [comparisonText, setComparisonText] = useState<string>("");
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    // Check cache first
    const cached = comparisonCache.current.get(candidate.id);
    if (cached) {
      setComparisonText(cached);
      return;
    }

    async function runComparison() {
      setComparisonLoading(true);
      setComparisonError(null);

      const userMessages = getUserMessages();
      if (userMessages.length === 0) {
        setComparisonError("Ingen brugerbeskeder at sammenligne med.");
        setComparisonLoading(false);
        return;
      }

      // Build answers section
      const answersSection = candidate.answers
        ? Object.entries(candidate.answers)
            .filter(([key]) => questions[key])
            .map(([key, ans]) => {
              const label = SCORE_LABELS[ans.score] ?? `${ans.score}`;
              const comment = ans.comment ? ` — "${ans.comment}"` : "";
              return `- ${questions[key]}: ${label}${comment}`;
            })
            .join("\n")
        : "Ingen svar tilgængelige.";

      const prompt = `Du er en neutral politisk analytiker. Sammenlign brugerens holdninger med denne kandidats svar og prioriteter.

Brugerens udtalelser fra samtalen:
${userMessages.join("\n\n")}

Kandidat: ${candidate.name} (${candidate.party})
Alder: ${candidate.age ?? "Ukendt"}
Stilling: ${candidate.occupation ?? "Ukendt"}

Kandidatens prioriteter: ${candidate.priorities?.join(", ") ?? "Ingen angivet"}

Kandidatens svar på politiske spørgsmål:
${answersSection}

Skriv 3-5 korte punkter på dansk om de vigtigste enigheder og uenigheder mellem brugeren og kandidaten. Fokuser på det mest relevante baseret på hvad brugeren faktisk har udtalt sig om. Hold det kort, neutralt og konkret.`;

      try {
        const controller = new AbortController();
        abortRef.current = controller;

        const model = createModel(provider);
        const result = streamText({
          model,
          prompt,
          abortSignal: controller.signal,
          maxOutputTokens: 1024,
          temperature: 0.5,
        });

        let accumulated = "";
        for await (const chunk of result.textStream) {
          accumulated += chunk;
          setComparisonText(accumulated);
        }

        // Cache the final result
        comparisonCache.current.set(candidate.id, accumulated);
      } catch (e: any) {
        if (e?.name !== "AbortError") {
          console.error("[CARD] Comparison failed:", e);
          setComparisonError("Kunne ikke generere sammenligning.");
        }
      } finally {
        setComparisonLoading(false);
      }
    }

    runComparison();

    return () => {
      abortRef.current?.abort();
    };
  }, [candidate.id]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-3">
      {/* Profile section */}
      <div>
        <h3 className="font-semibold text-gray-900 text-base">{candidate.name}</h3>
        <div className="text-sm text-gray-600 space-y-0.5">
          {candidate.age && <span>Alder: {candidate.age}</span>}
          {candidate.age && candidate.occupation && <span> · </span>}
          {candidate.occupation && <span>{candidate.occupation}</span>}
        </div>
        <p className="text-sm text-gray-500">{candidate.party}</p>
      </div>

      {candidate.pitch && (
        <p className="text-sm text-gray-700 italic">{candidate.pitch}</p>
      )}

      {candidate.priorities && candidate.priorities.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {candidate.priorities.map((p, i) => (
            <span
              key={i}
              className="text-xs bg-red-50 text-red-700 px-2 py-0.5 rounded-full"
            >
              {p}
            </span>
          ))}
        </div>
      )}

      {/* LLM comparison section */}
      <div className="border-t pt-3">
        <h4 className="text-sm font-medium text-gray-700 mb-2">
          Sammenligning med dine holdninger
        </h4>
        {comparisonError ? (
          <p className="text-sm text-red-600">{comparisonError}</p>
        ) : comparisonText ? (
          <div className="text-sm text-gray-700 whitespace-pre-wrap">{comparisonText}</div>
        ) : comparisonLoading ? (
          <div className="text-sm text-gray-400 animate-pulse">Genererer sammenligning...</div>
        ) : null}
      </div>
    </div>
  );
}
