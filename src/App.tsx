import { useRef, useState, useEffect, useCallback } from "react";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  ThreadPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  useMessagePartText,
  useThreadRuntime,
} from "@assistant-ui/react";
import { createAISDKAdapter, type UsageInfo } from "./adapters/ai-sdk-adapter";
import { EmbeddingManager } from "./lib/embeddings";
import { buildApiPrompt } from "./lib/system-prompt";
import type { Provider } from "./lib/provider";
import { ProviderSelector } from "./components/ProviderSelector";
import { StorkredsSelector } from "./components/StorkredsSelector";
import { ResultsPage } from "./components/ResultsPage";
import { ChoiceToolUI } from "./components/ChoiceButtons";

interface Storkreds {
  name: string;
  candidateCount: number;
}

interface CandidateAnswer {
  score: number;
  comment: string;
}

export interface CandidateData {
  id: number;
  name: string;
  party: string;
  partyLetter: string;
  area: string;
  age?: number;
  occupation?: string;
  pitch?: string;
  priorities?: string[];
  answers?: Record<string, CandidateAnswer>;
}

// --- Thread UI built from assistant-ui primitives ---

const THINKING_PREFIX = "💭\n";

function ThinkingBlock({ text, streaming }: { text: string; streaming: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const hasError = text.includes("\n---\n");
  const thinkingContent = hasError ? text.split("\n---\n")[0] : text;
  const errorMessage = hasError ? text.split("\n---\n").slice(1).join("\n---\n").trim() : null;

  return (
    <div>
      <div
        className="text-gray-400 italic text-sm cursor-pointer select-none flex items-center gap-1 hover:text-gray-500"
        onClick={() => setExpanded(!expanded)}
      >
        <span className={streaming && !hasError ? "animate-pulse" : ""}>
          {hasError ? "Tænkte..." : "Tænker..."}
        </span>
        <span className="text-xs">{expanded ? "▾" : "▸"}</span>
      </div>
      {expanded && (
        <pre className="text-xs text-gray-400 mt-1 whitespace-pre-wrap max-h-48 overflow-y-auto border-l-2 border-gray-200 pl-2 font-sans">
          {thinkingContent}
        </pre>
      )}
      {errorMessage && (
        <span className="block mt-2 text-gray-600 text-sm">{errorMessage}</span>
      )}
    </div>
  );
}

function TextPart() {
  const part = useMessagePartText();

  const thinkingIdx = part.text.indexOf(THINKING_PREFIX);
  if (thinkingIdx !== -1) {
    const beforeThinking = part.text.slice(0, thinkingIdx).trim();
    const thinkingContent = part.text.slice(thinkingIdx + THINKING_PREFIX.length);
    const isStreaming = !thinkingContent.includes("\n---\n");

    return (
      <>
        {beforeThinking && <span className="block mb-2">{beforeThinking}</span>}
        <ThinkingBlock text={thinkingContent} streaming={isStreaming} />
      </>
    );
  }

  return <span>{part.text}</span>;
}

function UserMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-end mb-4">
      <div className="bg-gray-100 text-gray-900 rounded-2xl rounded-br-sm px-4 py-2 max-w-[80%]">
        <MessagePrimitive.Content components={{ Text: TextPart }} />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-start mb-4">
      <div className="text-gray-900 max-w-[80%] px-1">
        <MessagePrimitive.Content components={{ Text: TextPart }} />
      </div>
    </MessagePrimitive.Root>
  );
}

/** Track the visual viewport height so the app fits above the iOS keyboard.
 *  Also prevents iOS Safari from scrolling the page when the keyboard opens. */
function useVisualViewportHeight() {
  const [height, setHeight] = useState(
    () => window.visualViewport?.height ?? window.innerHeight,
  );

  useEffect(() => {
    // Lock html/body to prevent iOS page-level scroll
    document.documentElement.style.overflow = "hidden";
    document.documentElement.style.height = "100%";
    document.body.style.overflow = "hidden";
    document.body.style.height = "100%";
    document.body.style.position = "fixed";
    document.body.style.width = "100%";

    const vv = window.visualViewport;
    if (!vv) return;

    const update = () => {
      setHeight(vv.height);
      // Force-reset any scroll iOS applied
      window.scrollTo(0, 0);
    };

    vv.addEventListener("resize", update);
    vv.addEventListener("scroll", update);
    return () => {
      vv.removeEventListener("resize", update);
      vv.removeEventListener("scroll", update);
    };
  }, []);

  return height;
}

function Composer({ requestCount }: { requestCount: number }) {
  return (
    <div className="border-t bg-white p-4 pb-2">
      <div className="max-w-3xl mx-auto w-full">
        <ComposerPrimitive.Root className="relative">
          <ComposerPrimitive.Input
            placeholder="Skriv din besked..."
            className="w-full resize-none rounded-2xl border border-gray-300 px-4 py-3 pr-12 text-base sm:text-sm outline-none focus:ring-2 focus:ring-ft-red-light focus:border-ft-red shadow-sm"
          />
          <ComposerPrimitive.Send className="absolute right-3 bottom-3 w-8 h-8 flex items-center justify-center rounded-full bg-ft-red text-white hover:bg-ft-red-dark disabled:opacity-30 disabled:bg-gray-300 transition">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 14V2M8 2L3 7M8 2L13 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </ComposerPrimitive.Send>
        </ComposerPrimitive.Root>
        <StatusNote requestCount={requestCount} />
      </div>
    </div>
  );
}

// --- Welcome screen with suggestion cards ---

const SUGGESTIONS = [
  "Hvad mener du om forsvarsudgifter?",
  "Hvad tænker du om klimapolitik?",
  "Fortæl om din holdning til skat og ulighed",
  "Hvad er vigtigst for dig ved dette valg?",
];

const TOTAL_QUESTIONS = 24;
// Gemini free tier: 1,500 requests/day
const GEMINI_FREE_TIER_RPD = 1500;

// --- Subtle status note shown below composer ---

function StatusNote({ requestCount }: { requestCount: number }) {
  const threadRuntime = useThreadRuntime();
  const [questionCount, setQuestionCount] = useState(0);

  useEffect(() => {
    return threadRuntime.subscribe(() => {
      const msgs = threadRuntime.getState().messages ?? [];
      const userMsgs = msgs.filter((m: any) => m.role === "user").length;
      setQuestionCount(userMsgs);
    });
  }, [threadRuntime]);

  const budgetPct = Math.min((requestCount / GEMINI_FREE_TIER_RPD) * 100, 100);

  if (questionCount === 0) return null;

  return (
    <div className="text-[10px] text-gray-400 text-right pr-1 pt-0.5">
      Spørgsmål {questionCount}/{TOTAL_QUESTIONS} · {budgetPct.toFixed(1)}% brugt
    </div>
  );
}

function WelcomeScreen({ hasProvider, onOpenSettings }: { hasProvider: boolean; onOpenSettings: () => void }) {
  const threadRuntime = useThreadRuntime();

  return (
    <div className="flex flex-col items-center justify-center h-full px-4">
      <h2 className="text-2xl font-bold text-ft-red-dark mb-2">
        Velkommen til Kandidattesten
      </h2>
      <p className="text-gray-500 mb-2 text-center max-w-md">
        Fortæl mig om dine politiske holdninger, så finder vi dine kandidater
      </p>
      <p className="text-gray-400 mb-8 text-center max-w-md text-sm">
        Du behøver ikke svare på alle spørgsmål — fortæl om det der er vigtigst for dig og gå til Resultater når du vil
      </p>

      {!hasProvider ? (
        <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 max-w-md w-full text-center">
          <p className="text-gray-600 text-sm mb-3">
            Vælg en AI-udbyder for at komme i gang
          </p>
          <button
            onClick={onOpenSettings}
            className="px-5 py-2.5 bg-ft-red text-white rounded-lg text-sm font-medium hover:bg-ft-red-dark transition"
          >
            Kom i gang
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg w-full">
          {SUGGESTIONS.map((text) => (
            <button
              key={text}
              onClick={() => {
                threadRuntime.append({
                  role: "user",
                  content: [{ type: "text", text }],
                });
              }}
              className="text-left p-4 rounded-xl border border-gray-200 text-sm text-gray-700 hover:bg-gray-50 hover:border-gray-300 transition"
            >
              {text}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function ChatThread({ hasProvider, onOpenSettings, requestCount }: { hasProvider: boolean; onOpenSettings: () => void; requestCount: number }) {
  return (
    <ThreadPrimitive.Root className="flex flex-col h-full">
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto w-full px-4 py-4">
          <ThreadPrimitive.Empty>
            <WelcomeScreen hasProvider={hasProvider} onOpenSettings={onOpenSettings} />
          </ThreadPrimitive.Empty>
          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              AssistantMessage,
            }}
          />
        </div>
        <ThreadPrimitive.ViewportFooter />
      </ThreadPrimitive.Viewport>
      <Composer requestCount={requestCount} />
    </ThreadPrimitive.Root>
  );
}

// --- User message collector ---

function UserMessageCollector({ messagesRef }: { messagesRef: React.MutableRefObject<() => string[]> }) {
  const threadRuntime = useThreadRuntime();
  messagesRef.current = () => {
    const state = threadRuntime.getState();
    return (state.messages ?? [])
      .filter((m: any) => m.role === "user")
      .map((m: any) =>
        m.content
          .filter((p: any) => p.type === "text")
          .map((p: any) => p.text)
          .join(""),
      );
  };
  return null;
}

// --- Stub adapter for no-provider state ---

function createStubAdapter(message?: string) {
  return {
    async *run() {
      yield {
        content: [
          {
            type: "text" as const,
            text: message ?? "Konfigurer en model under Indstillinger for at starte chatten.",
          },
        ],
      };
    },
  };
}

// --- Session persistence helpers ---

const STORAGE_KEY = "ktest_saved_chat";

function saveSession(messages: Array<{ role: string; content: any[] }>) {
  const serializable = messages.map((m) => ({
    role: m.role,
    content: m.content
      .filter((p: any) => p.type === "text")
      .map((p: any) => ({ type: "text" as const, text: p.text })),
  }));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
}

function loadSession(): Array<{ role: "user" | "assistant"; content: string }> | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function clearSession() {
  localStorage.removeItem(STORAGE_KEY);
}

function hasSavedSession() {
  return localStorage.getItem(STORAGE_KEY) !== null;
}

// --- Save / Load / Clear session buttons ---

function SaveSessionButton() {
  const threadRuntime = useThreadRuntime();
  const [saved, setSaved] = useState(false);
  const [hasExisting, setHasExisting] = useState(hasSavedSession);

  const handleSave = () => {
    const state = threadRuntime.getState();
    const msgs = state.messages ?? [];
    if (msgs.length === 0) return;
    saveSession(msgs as any);
    setSaved(true);
    setHasExisting(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleLoad = () => {
    const msgs = loadSession();
    if (!msgs) return;
    threadRuntime.reset(
      msgs.map((m) => ({
        role: m.role as "user" | "assistant",
        content: typeof m.content === "string" ? m.content : m.content,
      })),
    );
  };

  const handleClear = () => {
    clearSession();
    setHasExisting(false);
  };

  return (
    <div className="flex items-center gap-1">
      <button
        onClick={handleSave}
        className="text-xs text-gray-500 hover:text-ft-red-dark transition whitespace-nowrap"
        title="Gem samtale til browser"
      >
        {saved ? "Gemt ✓" : "Gem Samtale"}
      </button>
      {hasExisting && (
        <>
          <button
            onClick={handleLoad}
            className="text-xs text-gray-500 hover:text-ft-red-dark transition whitespace-nowrap"
            title="Indlæs gemt samtale"
          >
            Indlæs
          </button>
          <button
            onClick={handleClear}
            className="text-xs text-gray-400 hover:text-red-500 transition"
            title="Slet gemt samtale"
          >
            ✕
          </button>
        </>
      )}
    </div>
  );
}

// --- Settings gear icon ---

function SettingsButton({ hasProvider, onClick }: { hasProvider: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="relative p-2 text-gray-500 hover:text-ft-red-dark transition"
      title="Indstillinger"
    >
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="10" cy="10" r="3" />
        <path d="M10 1.5v2M10 16.5v2M1.5 10h2M16.5 10h2M3.4 3.4l1.4 1.4M15.2 15.2l1.4 1.4M3.4 16.6l1.4-1.4M15.2 4.8l1.4-1.4" />
      </svg>
      {!hasProvider && (
        <span className="absolute top-1 right-1 w-2.5 h-2.5 bg-ft-red rounded-full border-2 border-white" />
      )}
    </button>
  );
}

// --- Main App ---

export default function App() {
  const vpHeight = useVisualViewportHeight();
  const embeddingsRef = useRef<EmbeddingManager | null>(null);
  const [provider, setProvider] = useState<Provider | null>(null);
  const [modelsReady, setModelsReady] = useState(false);
  const [selectedStorkreds, setSelectedStorkreds] = useState<string | null>(null);
  const [storkredse, setStorkredse] = useState<Storkreds[]>([]);
  const [candidates, setCandidates] = useState<Map<number, CandidateData>>(new Map());
  const [systemPrompt, setSystemPrompt] = useState<string | null>(null);
  const [screen, setScreen] = useState<"chat" | "results">("chat");
  const [questions, setQuestions] = useState<Record<string, string>>({});
  const [showSettings, setShowSettings] = useState(false);
  const [requestCount, setRequestCount] = useState(0);
  const getUserMessagesRef = useRef<() => string[]>(() => []);

  const handleProgress = useCallback(
    (data: { status?: string; file?: string; progress?: number }) => {
      if (!data.file) return;
    },
    [],
  );

  // Load static data on mount
  useEffect(() => {
    fetch("/storkredse.json")
      .then((r) => r.json())
      .then(setStorkredse);

    fetch("/candidates.json")
      .then((r) => r.json())
      .then((data: CandidateData[]) => {
        setCandidates(new Map(data.map((c) => [c.id, c])));
      });
  }, []);

  // Load embedding model when provider is selected
  useEffect(() => {
    if (!provider) return;

    async function loadModels() {
      try {
        const backend = provider!.kind === "google" ? "google-api" as const : "wasm" as const;
        embeddingsRef.current = new EmbeddingManager({
          backend,
          onProgress: handleProgress,
          apiKey: provider!.apiKey,
        });
        await embeddingsRef.current.load();
        console.log("[APP] Embedding model loaded (backend: %s)", backend);
      } catch (e) {
        console.error("[APP] Embedding model failed:", e);
      }
      setModelsReady(true);
    }

    loadModels();

    return () => {
      embeddingsRef.current?.dispose();
    };
  }, [provider, handleProgress]);

  // Load system prompt when provider changes or storkreds changes
  useEffect(() => {
    if (!provider) return;

    async function loadPrompt() {
      const [national, allLocal] = await Promise.all([
        fetch("/questions.json").then((r) => r.json()),
        fetch("/local_questions.json").then((r) => r.json()),
      ]);
      const localMap = allLocal as Record<string, Record<string, string>>;

      let localForStorkreds: Record<string, string> = {};
      if (selectedStorkreds) {
        const storkredsSlug = selectedStorkreds
          .replace(/\s*storkreds$/i, "")
          .toLowerCase()
          .replace(/\s+/g, "-");
        localForStorkreds = localMap[storkredsSlug] ?? {};
      }

      setQuestions(national);
      setSystemPrompt(
        buildApiPrompt(national, localForStorkreds, selectedStorkreds),
      );
    }

    loadPrompt();
  }, [selectedStorkreds, provider]);

  const handleProviderSelect = useCallback((p: Provider) => {
    setProvider(p);
  }, []);

  const handleUsage = useCallback((usage: UsageInfo) => {
    setRequestCount((c) => c + 1);
    console.log("[APP] Token usage:", usage);
  }, []);

  const adapter = (() => {
    if (!provider) return createStubAdapter();
    return createAISDKAdapter(provider, systemPrompt ?? "", handleUsage);
  })();

  const runtime = useLocalRuntime(adapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div
        className="fixed inset-x-0 top-0 flex flex-col bg-white overflow-hidden"
        style={{ height: `${vpHeight}px` }}
      >
        {/* Header */}
        <header className="border-b bg-white shrink-0">
          <div className="max-w-5xl mx-auto w-full px-4 py-2 flex items-center gap-3">
            {/* Left: Storkreds dropdown */}
            <StorkredsSelector
              storkredse={storkredse}
              selected={selectedStorkreds}
              onSelect={setSelectedStorkreds}
            />

            {/* Center: Title */}
            <h1 className="text-lg font-bold text-ft-red-dark flex-1 text-center">
              Kandidattest
            </h1>

            {/* Right: Provider info + Save + Settings */}
            <div className="flex items-center gap-2">
              {provider && (
                <span className="text-xs text-gray-400 hidden sm:inline">
                  {`${provider.label}: ${provider.model}`}
                </span>
              )}
              <SaveSessionButton />
              <SettingsButton
                hasProvider={!!provider}
                onClick={() => setShowSettings(true)}
              />
            </div>
          </div>

          {/* Tab bar */}
          <div className="max-w-5xl mx-auto w-full px-4">
            <nav className="flex gap-6">
              <button
                onClick={() => setScreen("chat")}
                className={`text-sm pb-2 border-b-2 transition font-medium ${
                  screen === "chat"
                    ? "border-ft-red text-ft-red-dark"
                    : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
              >
                Samtale
              </button>
              <button
                onClick={() => setScreen("results")}
                className={`text-sm pb-2 border-b-2 transition font-medium ${
                  screen === "results"
                    ? "border-ft-red text-ft-red-dark"
                    : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
              >
                Resultater
              </button>
            </nav>
          </div>
        </header>

        {/* Main content */}
        {screen === "chat" ? (
          <div className="flex-1 overflow-hidden">
            <ChatThread
              hasProvider={!!provider}
              onOpenSettings={() => setShowSettings(true)}
              requestCount={requestCount}
            />
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <ResultsPage
              embeddingsRef={embeddingsRef}
              modelsReady={modelsReady}
              selectedStorkreds={selectedStorkreds}
              candidates={candidates}
              provider={provider!}
              questions={questions}
              getUserMessages={getUserMessagesRef.current}
            />
          </div>
        )}
      </div>

      {/* Modal */}
      <ProviderSelector
        open={showSettings}
        onClose={() => setShowSettings(false)}
        onSelect={handleProviderSelect}
      />
      <ChoiceToolUI />
      <UserMessageCollector messagesRef={getUserMessagesRef} />
    </AssistantRuntimeProvider>
  );
}
