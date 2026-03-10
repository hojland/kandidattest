import { useState, useEffect, useCallback, useRef, useMemo } from "react";
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
  h: "#666666",
  i: "#00B0CA",
  k: "#F29D00",
  m: "#532380",
  o: "#005078",
  v: "#004A98",
  æ: "#0F4D3A",
  ø: "#E5001A",
  å: "#2B8738",
};

const PARTY_NAMES: Record<string, string> = {
  a: "Socialdemokratiet",
  b: "Radikale Venstre",
  c: "Det Konservative Folkeparti",
  f: "SF",
  h: "Borgernes Parti",
  i: "Liberal Alliance",
  m: "Moderaterne",
  o: "Dansk Folkeparti",
  v: "Venstre",
  æ: "Danmarksdemokraterne",
  ø: "Enhedslisten",
  å: "Alternativet",
};

const PAGE_SIZES = [10, 20, 50];

interface Props {
  embeddingsRef: React.RefObject<EmbeddingManager | null>;
  modelsReady: boolean;
  selectedStorkreds: string | null;
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
  const [allMatches, setAllMatches] = useState<CandidateMatch[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const comparisonCacheRef = useRef<Map<number, string>>(new Map());

  // Filters & pagination
  const [search, setSearch] = useState("");
  const [partyFilter, setPartyFilter] = useState<string | null>(null);
  const [pageSize, setPageSize] = useState(10);
  const [page, setPage] = useState(0);

  const runMatch = useCallback(async () => {
    if (!embeddingsRef.current || !modelsReady) {
      setError("Embedding-modellen er ikke klar endnu.");
      return;
    }

    const userMessages = getUserMessages();
    if (userMessages.length === 0) {
      setError(
        "Du har ikke skrevet nogen beskeder endnu. Gå tilbage til samtalen og fortæl om dine politiske holdninger.",
      );
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const userText = userMessages.join("\n\n");
      const embedding = await embeddingsRef.current.embed(userText);
      const results = embeddingsRef.current.findMatches(
        embedding,
        0, // return all
        selectedStorkreds ?? undefined,
      );
      setAllMatches(results);
      setPage(0);
    } catch (e) {
      console.error("[RESULTS] Match failed:", e);
      setError("Kunne ikke beregne matches. Prøv igen.");
    } finally {
      setLoading(false);
    }
  }, [embeddingsRef, modelsReady, selectedStorkreds, getUserMessages]);

  // Run match when embedding model becomes ready
  useEffect(() => {
    if (modelsReady) runMatch();
  }, [modelsReady]); // eslint-disable-line react-hooks/exhaustive-deps

  // Derive available parties from matches
  const availableParties = useMemo(() => {
    if (!allMatches) return [];
    const seen = new Set<string>();
    for (const m of allMatches) {
      const c = candidates.get(m.id);
      if (c?.partyLetter) seen.add(c.partyLetter.toLowerCase());
    }
    return Array.from(seen).sort();
  }, [allMatches, candidates]);

  // Filtered + searched results
  const filtered = useMemo(() => {
    if (!allMatches) return [];
    return allMatches.filter((m) => {
      const c = candidates.get(m.id);
      if (partyFilter && c?.partyLetter?.toLowerCase() !== partyFilter)
        return false;
      if (search) {
        const q = search.toLowerCase();
        const nameMatch = m.name.toLowerCase().includes(q);
        const partyMatch = c?.party?.toLowerCase().includes(q);
        if (!nameMatch && !partyMatch) return false;
      }
      return true;
    });
  }, [allMatches, candidates, partyFilter, search]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalPages - 1);
  const pageMatches = filtered.slice(
    safePage * pageSize,
    safePage * pageSize + pageSize,
  );

  // Reset page when filters change
  useEffect(() => {
    setPage(0);
  }, [search, partyFilter, pageSize]);

  if (!modelsReady || loading) {
    return (
      <div className="p-8 text-center">
        <div className="animate-pulse text-gray-500">
          {!modelsReady ? "Indlæser embedding-model..." : "Beregner matches..."}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <p className="text-gray-600 mb-4">{error}</p>
        <button
          onClick={runMatch}
          className="px-4 py-2 bg-ft-red text-white rounded-full text-sm font-medium hover:bg-ft-red-dark"
        >
          Prøv igen
        </button>
      </div>
    );
  }

  if (!allMatches || allMatches.length === 0) {
    return (
      <div className="p-8 text-center">
        <p className="text-gray-500">Ingen matches fundet.</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-gray-900">
          Dine matches ({filtered.length} kandidater)
        </h2>
        <button
          onClick={runMatch}
          className="px-3 py-1.5 text-xs bg-ft-red text-white rounded-full font-medium hover:bg-ft-red-dark"
        >
          Opdater matches
        </button>
      </div>

      {/* Search + filters */}
      <div className="flex flex-col sm:flex-row gap-2">
        <input
          type="text"
          placeholder="Søg efter navn eller parti..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 px-3 py-1.5 border rounded-lg text-sm outline-none focus:ring-2 focus:ring-ft-red-light"
        />
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => setPartyFilter(null)}
            className={`px-2 py-1 rounded-full text-xs font-medium transition ${
              partyFilter === null
                ? "bg-gray-800 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            Alle
          </button>
          {availableParties.map((pl) => (
            <button
              key={pl}
              onClick={() =>
                setPartyFilter(partyFilter === pl ? null : pl)
              }
              className="px-2 py-1 rounded-full text-xs font-medium transition"
              style={{
                backgroundColor:
                  partyFilter === pl
                    ? PARTY_COLORS[pl] || "#666"
                    : "#f3f4f6",
                color: partyFilter === pl ? "white" : "#4b5563",
              }}
              title={PARTY_NAMES[pl] || pl.toUpperCase()}
            >
              {pl.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Results list */}
      {pageMatches.length === 0 ? (
        <p className="text-center text-gray-400 py-4 text-sm">
          Ingen kandidater matcher din søgning.
        </p>
      ) : (
        pageMatches.map((match) => {
          const c = candidates.get(match.id);
          const color =
            PARTY_COLORS[c?.partyLetter?.toLowerCase() ?? ""] || "#666";
          const isExpanded = expandedId === match.id;
          // Global rank (1-indexed position in full sorted list)
          const rank = allMatches!.indexOf(match) + 1;

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
                  {rank}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline gap-2">
                    <span className="font-semibold text-gray-900">
                      {match.name}
                    </span>
                    <span className="text-sm text-gray-500">
                      ({c?.partyLetter?.toUpperCase()})
                    </span>
                    <span className="ml-auto text-sm font-medium text-ft-red-dark">
                      {Math.round(match.score * 100)}%
                    </span>
                  </div>
                  {c?.occupation && (
                    <p className="text-sm text-gray-500">{c.occupation}</p>
                  )}
                  {c?.area && (
                    <p className="text-xs text-gray-400">{c.area}</p>
                  )}
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
        })
      )}

      {/* Pagination controls */}
      {filtered.length > 0 && (
        <div className="flex items-center justify-between pt-2 border-t">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span>Vis</span>
            {PAGE_SIZES.map((size) => (
              <button
                key={size}
                onClick={() => setPageSize(size)}
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  pageSize === size
                    ? "bg-ft-red text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {size}
              </button>
            ))}
            <span>per side</span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setPage(Math.max(0, safePage - 1))}
              disabled={safePage === 0}
              className="px-2 py-1 rounded text-sm text-gray-600 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              ‹
            </button>
            <span className="text-sm text-gray-600 px-2">
              {safePage + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage(Math.min(totalPages - 1, safePage + 1))}
              disabled={safePage >= totalPages - 1}
              className="px-2 py-1 rounded text-sm text-gray-600 hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              ›
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
