export type ProviderKind = "openai" | "anthropic" | "google" | "openai-compatible";

export interface Provider {
  kind: ProviderKind;
  baseUrl?: string;   // required for openai-compatible, optional override for others
  apiKey: string;
  model: string;
  label: string;      // display name
}

export interface ProviderPreset {
  label: string;
  kind: ProviderKind;
  baseUrl?: string;
  defaultModel: string;
  keyPlaceholder: string;
}

export const API_PRESETS: ProviderPreset[] = [
  { label: "Gemini", kind: "google", defaultModel: "gemini-3-flash-preview", keyPlaceholder: "AIza..." },
  { label: "Anthropic", kind: "anthropic", defaultModel: "claude-sonnet-4-6-20250514", keyPlaceholder: "sk-ant-..." },
  { label: "OpenAI", kind: "openai", defaultModel: "gpt-4.1-mini", keyPlaceholder: "sk-..." },
  { label: "Mistral", kind: "openai-compatible", baseUrl: "https://api.mistral.ai/v1", defaultModel: "mistral-medium-latest", keyPlaceholder: "..." },
  { label: "Groq", kind: "openai-compatible", baseUrl: "https://api.groq.com/openai/v1", defaultModel: "llama-3.3-70b-versatile", keyPlaceholder: "gsk_..." },
  { label: "Together", kind: "openai-compatible", baseUrl: "https://api.together.xyz/v1", defaultModel: "meta-llama/Llama-4-Scout-17B-16E-Instruct", keyPlaceholder: "..." },
  { label: "Ollama", kind: "openai-compatible", baseUrl: "http://localhost:11434/v1", defaultModel: "qwen3:4b", keyPlaceholder: "ollama (anything)" },
];

/** Store provider config in sessionStorage (opt-in). Never localStorage. */
export function storeProviderSession(provider: Provider): void {
  sessionStorage.setItem("ktest_provider", JSON.stringify(provider));
}

/** Retrieve provider config from sessionStorage, or null. */
export function loadProviderSession(): Provider | null {
  const raw = sessionStorage.getItem("ktest_provider");
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (parsed.kind && parsed.model && parsed.apiKey) return parsed as Provider;
  } catch { /* ignore */ }
  return null;
}

/** Clear stored session. */
export function clearProviderSession(): void {
  sessionStorage.removeItem("ktest_provider");
}
