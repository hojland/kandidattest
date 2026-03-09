import { useState, useEffect, useCallback, useRef } from "react";
import type { CandidateData } from "../App";
import type { CandidateMatch } from "../lib/embeddings";
import type { EmbeddingManager } from "../lib/embeddings";
import type { Provider } from "../lib/provider";
import { CandidateCard } from "./CandidateCard";

const PARTY_COLORS: Record<string, string> = {
  a: "#A82721",
  b: "#733280",
  c: "#00583C",
  d: "#00505C",
  f: "#E4007F",
  i: "#00B0CA",
  k: "#F29D00",
  m: "#532380",
  o: "#005078",
  v: "#004A98",
  ø: "#E5001A",
  å: "#2B8738",
};

interface Props {
  embeddingsRef: React.RefObject<EmbeddingManager | null>;
  modelsReady: boolean;
  selectedStorkreds: string;
  candidates: Map<number, CandidateData>;
  provider: Provider;
  questions: Record<string, string>;
  getUserMessages: () => string[];
}

export function ResultsPage({
  embeddingsRef,
  modelsReady,
  selectedStorkreds,
  candidates,
  provider,
  questions,
  getUserMessages,
}: Props) {
  const [matches, setMatches] = useState<CandidateMatch[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const comparisonCacheRef = useRef<Map<number, string>>(new Map());

  const runMatch = useCallback(async () => {
    if (!embeddingsRef.current || !modelsReady) {
      setError("Embedding-modellen er ikke klar endnu.");
      return;
    }

    const userMessages = getUserMessages();
    if (userMessages.length === 0) {
      setError("Du har ikke skrevet nogen beskeder endnu. Gå tilbage til samtalen og fortæl om dine politiske holdninger.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const userText = userMessages.join("\n\n");
      const embedding = await embeddingsRef.current.embed(userText);
      const results = embeddingsRef.current.findMatches(
        embedding,
        10,
        selectedStorkreds,
      );
      setMatches(results);
    } catch (e) {
      console.error("[RESULTS] Match failed:", e);
      setError("Kunne ikke beregne matches. Prøv igen.");
    } finally {
      setLoading(false);
    }
  }, [embeddingsRef, modelsReady, selectedStorkreds, getUserMessages]);

  // Run match on mount
  useEffect(() => {
    runMatch();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) {
    return (
      <div className="p-8 text-center">
        <div className="animate-pulse text-gray-500">Beregner matches...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <p className="text-gray-600 mb-4">{error}</p>
        <button
          onClick={runMatch}
          className="px-4 py-2 bg-red-600 text-white rounded-full text-sm font-medium hover:bg-red-700"
        >
          Prøv igen
        </button>
      </div>
    );
  }

  if (!matches || matches.length === 0) {
    return (
      <div className="p-8 text-center">
        <p className="text-gray-500">Ingen matches fundet.</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-gray-900">Dine top-matches</h2>
        <button
          onClick={runMatch}
          className="px-3 py-1.5 text-xs bg-red-600 text-white rounded-full font-medium hover:bg-red-700"
        >
          Opdater matches
        </button>
      </div>
      {matches.map((match, i) => {
        const c = candidates.get(match.id);
        const color =
          PARTY_COLORS[c?.partyLetter?.toLowerCase() ?? ""] || "#666";
        const isExpanded = expandedId === match.id;

        return (
          <div key={match.id}>
            <div
              onClick={() => setExpandedId(isExpanded ? null : match.id)}
              className="flex items-start gap-3 p-3 rounded-lg border bg-white shadow-sm cursor-pointer hover:bg-gray-50 transition"
            >
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm shrink-0"
                style={{ backgroundColor: color }}
              >
                {i + 1}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-2">
                  <span className="font-semibold text-gray-900">
                    {match.name}
                  </span>
                  <span className="text-sm text-gray-500">
                    ({c?.partyLetter?.toUpperCase()})
                  </span>
                  <span className="ml-auto text-sm font-medium text-red-700">
                    {Math.round(match.score * 100)}%
                  </span>
                </div>
                {c?.occupation && (
                  <p className="text-sm text-gray-500">{c.occupation}</p>
                )}
                {c?.area && <p className="text-xs text-gray-400">{c.area}</p>}
              </div>
              <span className="text-gray-400 text-xs mt-1">
                {isExpanded ? "▾" : "▸"}
              </span>
            </div>
            {isExpanded && c && (
              <div className="mt-1 ml-11 border rounded-lg bg-gray-50 p-4">
                <CandidateCard
                  candidate={c}
                  provider={provider}
                  questions={questions}
                  getUserMessages={getUserMessages}
                  comparisonCache={comparisonCacheRef}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
