interface Storkreds {
  name: string;
  candidateCount: number;
}

interface Props {
  storkredse: Storkreds[];
  selected: string | null;
  onSelect: (s: string | null) => void;
}

export function StorkredsSelector({ storkredse, selected, onSelect }: Props) {
  return (
    <select
      value={selected ?? ""}
      onChange={(e) => onSelect(e.target.value || null)}
      className="rounded-full bg-gray-100 border-none px-3 py-1.5 text-sm text-gray-700 outline-none cursor-pointer hover:bg-gray-200 transition"
    >
      <option value="">Hele Danmark</option>
      {storkredse.map((s) => (
        <option key={s.name} value={s.name}>
          {s.name}
        </option>
      ))}
    </select>
  );
}
