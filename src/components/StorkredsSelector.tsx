interface Storkreds {
  name: string;
  candidateCount: number;
}

interface Props {
  storkredse: Storkreds[];
  onSelect: (storkreds: string) => void;
}

export function StorkredsSelector({ storkredse, onSelect }: Props) {
  return (
    <div className="flex flex-col items-center justify-center h-full p-6">
      <h1 className="text-2xl font-bold text-red-700 mb-2">Kandidattest</h1>
      <p className="text-gray-600 mb-6 text-center max-w-md">
        Chat med AI om politik og find de kandidater der passer bedst til dine
        holdninger.
      </p>
      <label className="text-sm font-medium text-gray-700 mb-2">
        Vælg din storkreds
      </label>
      <select
        className="w-72 p-3 border rounded-lg text-gray-900 bg-white shadow-sm"
        defaultValue=""
        onChange={(e) => e.target.value && onSelect(e.target.value)}
      >
        <option value="" disabled>
          — Vælg storkreds —
        </option>
        {storkredse.map((s) => (
          <option key={s.name} value={s.name}>
            {s.name} ({s.candidateCount} kandidater)
          </option>
        ))}
      </select>
      <p className="text-xs text-gray-400 mt-3">
        Din storkreds bestemmer hvilke lokale spørgsmål der indgår i samtalen.
      </p>
    </div>
  );
}
