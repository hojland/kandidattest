import { useState, useEffect } from "react";
import {
  type Provider,
  API_PRESETS,
  loadApiSession,
  storeApiSession,
  clearApiSession,
} from "../lib/provider";

interface Props {
  webgpuSupported: boolean;
  onSelect: (provider: Provider) => void;
}

export function ProviderSelector({ webgpuSupported, onSelect }: Props) {
  const [showApiForm, setShowApiForm] = useState(false);
  const [preset, setPreset] = useState(0);
  const [baseUrl, setBaseUrl] = useState(API_PRESETS[0].baseUrl);
  const [model, setModel] = useState(API_PRESETS[0].defaultModel);
  const [apiKey, setApiKey] = useState("");
  const [remember, setRemember] = useState(false);

  // On mount, check for saved session
  useEffect(() => {
    const saved = loadApiSession();
    if (saved && saved.type === "api") {
      setShowApiForm(true);
      setBaseUrl(saved.baseUrl);
      setApiKey(saved.apiKey);
      setModel(saved.model);
      setRemember(true);
      // Try to match a preset
      const idx = API_PRESETS.findIndex((p) => p.baseUrl === saved.baseUrl);
      if (idx >= 0) setPreset(idx);
    }
  }, []);

  function handlePresetChange(index: number) {
    setPreset(index);
    setBaseUrl(API_PRESETS[index].baseUrl);
    setModel(API_PRESETS[index].defaultModel);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const provider: Provider = { type: "api", baseUrl, apiKey, model };
    if (remember) {
      storeApiSession(provider);
    } else {
      clearApiSession();
    }
    onSelect(provider);
  }

  const allFilled = baseUrl.trim() && apiKey.trim() && model.trim();

  if (!showApiForm) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-6">
        <div className="text-center max-w-md w-full">
          <h1 className="text-2xl font-bold text-red-700 mb-2">
            Kandidattest
          </h1>
          <p className="text-gray-500 mb-8">
            Vælg hvordan AI-modellen skal køre
          </p>

          <div className="flex flex-col gap-4">
            <button
              onClick={() => onSelect({ type: "local" })}
              disabled={!webgpuSupported}
              className="w-full rounded-xl border border-gray-200 p-5 text-left hover:border-red-300 hover:bg-red-50 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="font-semibold text-gray-900 mb-1">
                Lokal (WebGPU)
              </div>
              <p className="text-sm text-gray-500">
                Kører helt i browseren. Ingen data sendes nogen steder.
              </p>
              {!webgpuSupported && (
                <p className="text-xs text-red-600 mt-2">
                  (ikke understøttet i denne browser)
                </p>
              )}
            </button>

            <button
              onClick={() => setShowApiForm(true)}
              className="w-full rounded-xl border border-gray-200 p-5 text-left hover:border-red-300 hover:bg-red-50 transition"
            >
              <div className="font-semibold text-gray-900 mb-1">
                API (OpenAI-kompatibel)
              </div>
              <p className="text-sm text-gray-500">
                Brug OpenAI, Groq, Together, Ollama eller anden kompatibel API.
              </p>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center h-full p-6">
      <div className="max-w-md w-full">
        <h1 className="text-2xl font-bold text-red-700 mb-2 text-center">
          Kandidattest
        </h1>
        <p className="text-gray-500 mb-6 text-center">
          Konfigurer API-forbindelse
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

          {/* Base URL */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Base URL
            </label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.openai.com/v1"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-red-200 focus:border-red-300"
            />
          </div>

          {/* Model */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model
            </label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="gpt-4o-mini"
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
              placeholder="sk-..."
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

          {/* Buttons */}
          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={() => setShowApiForm(false)}
              className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
            >
              Tilbage
            </button>
            <button
              type="submit"
              disabled={!allFilled}
              className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Fortsæt
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
