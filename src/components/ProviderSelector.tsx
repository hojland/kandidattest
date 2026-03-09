import { useState, useEffect } from "react";
import {
  type Provider,
  API_PRESETS,
  loadProviderSession,
  storeProviderSession,
  clearProviderSession,
} from "../lib/provider";

interface Props {
  onSelect: (provider: Provider) => void;
}

export function ProviderSelector({ onSelect }: Props) {
  const [preset, setPreset] = useState(0);
  const [baseUrl, setBaseUrl] = useState(API_PRESETS[0].baseUrl ?? "");
  const [model, setModel] = useState(API_PRESETS[0].defaultModel);
  const [apiKey, setApiKey] = useState("");
  const [remember, setRemember] = useState(false);

  // On mount, check for saved session
  useEffect(() => {
    const saved = loadProviderSession();
    if (saved) {
      if (saved.baseUrl) setBaseUrl(saved.baseUrl);
      setApiKey(saved.apiKey);
      setModel(saved.model);
      setRemember(true);
      const idx = API_PRESETS.findIndex((p) => p.label === saved.label);
      if (idx >= 0) setPreset(idx);
    }
  }, []);

  function handlePresetChange(index: number) {
    setPreset(index);
    const p = API_PRESETS[index];
    setBaseUrl(p.baseUrl ?? "");
    setModel(p.defaultModel);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const p = API_PRESETS[preset];
    const provider: Provider = {
      kind: p.kind,
      baseUrl: p.kind === "openai-compatible" ? baseUrl : undefined,
      apiKey,
      model,
      label: p.label,
    };
    if (remember) {
      storeProviderSession(provider);
    } else {
      clearProviderSession();
    }
    onSelect(provider);
  }

  const currentPreset = API_PRESETS[preset];
  const needsBaseUrl = currentPreset.kind === "openai-compatible";
  const allFilled = apiKey.trim() && model.trim() && (!needsBaseUrl || baseUrl.trim());

  return (
    <div className="flex flex-col items-center justify-center h-full p-6">
      <div className="max-w-md w-full">
        <h1 className="text-2xl font-bold text-red-700 mb-2 text-center">
          Kandidattest
        </h1>
        <p className="text-gray-500 mb-6 text-center">
          Konfigurer AI-forbindelse
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Preset dropdown */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Udbyder
            </label>
            <select
              value={preset}
              onChange={(e) => handlePresetChange(Number(e.target.value))}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-red-200 focus:border-red-300"
            >
              {API_PRESETS.map((p, i) => (
                <option key={p.label} value={i}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          {/* Base URL — only for openai-compatible */}
          {needsBaseUrl && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Base URL
              </label>
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.example.com/v1"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-red-200 focus:border-red-300"
              />
            </div>
          )}

          {/* Model */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model
            </label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder={currentPreset.defaultModel}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-red-200 focus:border-red-300"
            />
          </div>

          {/* API key */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              API-nøgle
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={currentPreset.keyPlaceholder}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-red-200 focus:border-red-300"
            />
          </div>

          {/* Remember checkbox */}
          <div className="flex items-start gap-2">
            <input
              type="checkbox"
              id="remember"
              checked={remember}
              onChange={(e) => setRemember(e.target.checked)}
              className="mt-1 accent-red-600"
            />
            <label htmlFor="remember" className="text-sm text-gray-600">
              <span className="font-medium">Husk for denne session</span>
              <br />
              <span className="text-xs text-gray-400">
                Gemmes i sessionStorage og slettes når browseren lukkes.
              </span>
            </label>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={!allFilled}
            className="w-full rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Fortsæt
          </button>
        </form>
      </div>
    </div>
  );
}
