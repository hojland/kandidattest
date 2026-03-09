export type Provider =
  | { type: "local" }
  | { type: "api"; baseUrl: string; apiKey: string; model: string };

export const API_PRESETS: { label: string; baseUrl: string; defaultModel: string }[] = [
  { label: "OpenAI", baseUrl: "https://api.openai.com/v1", defaultModel: "gpt-4o-mini" },
  { label: "Groq", baseUrl: "https://api.groq.com/openai/v1", defaultModel: "llama-3.3-70b-versatile" },
  { label: "Together", baseUrl: "https://api.together.xyz/v1", defaultModel: "meta-llama/Llama-3.3-70B-Instruct-Turbo" },
  { label: "Ollama", baseUrl: "http://localhost:11434/v1", defaultModel: "llama3.2" },
];

/** Store API key in sessionStorage (opt-in). Never localStorage. */
export function storeApiSession(provider: Provider): void {
  if (provider.type !== "api") return;
  sessionStorage.setItem("ktest_provider", JSON.stringify({
    baseUrl: provider.baseUrl,
    apiKey: provider.apiKey,
    model: provider.model,
  }));
}

/** Retrieve API config from sessionStorage, or null. */
export function loadApiSession(): Provider | null {
  const raw = sessionStorage.getItem("ktest_provider");
  if (!raw) return null;
  try {
    const { baseUrl, apiKey, model } = JSON.parse(raw);
    if (baseUrl && apiKey && model) return { type: "api", baseUrl, apiKey, model };
  } catch { /* ignore */ }
  return null;
}

/** Clear stored session. */
export function clearApiSession(): void {
  sessionStorage.removeItem("ktest_provider");
}
