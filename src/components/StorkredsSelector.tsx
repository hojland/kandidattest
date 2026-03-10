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
      className="appearance-none rounded-full bg-gray-100 border-none px-3 pr-7 py-1.5 text-sm text-gray-700 outline-none cursor-pointer hover:bg-gray-200 transition bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236b7280%22%20stroke-width%3D%222.5%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22m6%209%206%206%206-6%22%2F%3E%3C%2Fsvg%3E')] bg-[length:12px] bg-[right_8px_center] bg-no-repeat"
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
