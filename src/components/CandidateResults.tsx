import type { CandidateMatch } from "../lib/embeddings";

interface CandidateData {
  id: number;
  name: string;
  party: string;
  partyLetter: string;
  area: string;
  age?: number;
  occupation?: string;
  pitch?: string;
}

interface Props {
  matches: CandidateMatch[];
  candidates: Map<number, CandidateData>;
}

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

export function CandidateResults({ matches, candidates }: Props) {
  return (
    <div className="p-4 space-y-3">
      <h2 className="text-lg font-bold text-gray-900">Dine top-matches</h2>
      {matches.map((match, i) => {
        const c = candidates.get(match.id);
        const color =
          PARTY_COLORS[c?.partyLetter?.toLowerCase() ?? ""] || "#666";
        return (
          <div
            key={match.id}
            className="flex items-start gap-3 p-3 rounded-lg border bg-white shadow-sm"
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
              {c?.pitch && (
                <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                  {c.pitch}
                </p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
