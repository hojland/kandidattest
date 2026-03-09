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
import { createAISDKAdapter } from "./adapters/ai-sdk-adapter";
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

  // Check for thinking content (💭\n prefix from adapters)
  const thinkingIdx = part.text.indexOf(THINKING_PREFIX);
  if (thinkingIdx !== -1) {
    const beforeThinking = part.text.slice(0, thinkingIdx).trim();
    const thinkingContent = part.text.slice(thinkingIdx + THINKING_PREFIX.length);
    // Streaming if the text doesn't contain the error marker
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
    <MessagePrimitive.Root className="flex justify-end mb-3">
      <div className="bg-red-50 text-gray-900 rounded-2xl rounded-br-sm px-4 py-2 max-w-[80%]">
        <MessagePrimitive.Content components={{ Text: TextPart }} />
      </div>
    </MessagePrimitive.Root>
  );
}

function AssistantMessage() {
  return (
    <MessagePrimitive.Root className="flex justify-start mb-3">
      <div className="bg-gray-100 text-gray-900 rounded-2xl rounded-bl-sm px-4 py-2 max-w-[80%]">
        <MessagePrimitive.Content components={{ Text: TextPart }} />
      </div>
    </MessagePrimitive.Root>
  );
}

function Composer() {
  return (
    <ComposerPrimitive.Root className="flex gap-2 p-3 border-t">
      <ComposerPrimitive.Input
        placeholder="Skriv din besked..."
        className="flex-1 px-4 py-2 border rounded-full text-sm outline-none focus:ring-2 focus:ring-red-200"
      />
      <ComposerPrimitive.Send className="px-4 py-2 bg-red-600 text-white rounded-full text-sm font-medium hover:bg-red-700 disabled:opacity-50">
        Send
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  );
}

function ChatThread() {
  return (
    <ThreadPrimitive.Root className="flex flex-col h-full">
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto p-4">
        <ThreadPrimitive.Empty>
          <p className="text-center text-gray-400 mt-8">
            Klar til at starte...
          </p>
        </ThreadPrimitive.Empty>
        <ThreadPrimitive.Messages
          components={{
            UserMessage,
            AssistantMessage,
          }}
        />
        <ThreadPrimitive.ViewportFooter />
      </ThreadPrimitive.Viewport>
      <Composer />
    </ThreadPrimitive.Root>
  );
}

// --- Auto-start: sends a hidden trigger to make the model speak first ---

function AutoStart({ ready }: { ready: boolean }) {
  const threadRuntime = useThreadRuntime();
  const started = useRef(false);

  useEffect(() => {
    if (!ready || started.current) return;
    started.current = true;
    threadRuntime.append({
      role: "user",
      content: [{ type: "text", text: "Hej! Start venligst samtalen." }],
    });
  }, [ready, threadRuntime]);

  return null;
}

// --- User message collector (reads messages from thread runtime) ---

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

// --- Main App ---

export default function App() {
  const embeddingsRef = useRef<EmbeddingManager | null>(null);
  const [provider, setProvider] = useState<Provider | null>(null);
  const [modelsReady, setModelsReady] = useState(false);
  const [selectedStorkreds, setSelectedStorkreds] = useState<string | null>(
    null,
  );
  const [storkredse, setStorkredse] = useState<Storkreds[]>([]);
  const [candidates, setCandidates] = useState<Map<number, CandidateData>>(
    new Map(),
  );
  const [systemPrompt, setSystemPrompt] = useState<string | null>(null);
  const [screen, setScreen] = useState<"chat" | "results">("chat");
  const [questions, setQuestions] = useState<Record<string, string>>({});
  const getUserMessagesRef = useRef<() => string[]>(() => []);

  const handleProgress = useCallback(
    (data: { status?: string; file?: string; progress?: number }) => {
      if (!data.file) return;
      // Progress tracking for embeddings if needed in the future
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
        embeddingsRef.current = new EmbeddingManager(handleProgress);
        await embeddingsRef.current.load();
        console.log("[APP] Embedding model loaded");
      } catch (e) {
        console.error("[APP] Embedding model failed:", e);
      }
      setModelsReady(true);
    }

    loadModels();
  }, [provider, handleProgress]);

  // Load system prompt when storkreds is selected
  useEffect(() => {
    if (!selectedStorkreds || !provider) return;

    async function loadPrompt() {
      const [national, allLocal] = await Promise.all([
        fetch("/questions.json").then((r) => r.json()),
        fetch("/local_questions.json").then((r) => r.json()),
      ]);
      const localMap = allLocal as Record<string, Record<string, string>>;
      const storkredsSlug = selectedStorkreds!
        .replace(/\s*storkreds$/i, "")
        .toLowerCase()
        .replace(/\s+/g, "-");
      const localForStorkreds = localMap[storkredsSlug] ?? {};

      setQuestions(national);
      setSystemPrompt(
        buildApiPrompt(national, localForStorkreds, selectedStorkreds!),
      );
    }

    loadPrompt();
  }, [selectedStorkreds, provider]);

  const handleProviderSelect = useCallback((p: Provider) => {
    setProvider(p);
  }, []);

  const adapter = provider
    ? createAISDKAdapter(provider, systemPrompt ?? "")
    : createAISDKAdapter(
        { kind: "openai", apiKey: "", model: "", label: "" },
        "",
      );

  const runtime = useLocalRuntime(adapter);

  // --- Render screens ---

  // Screen 1: Provider selection
  if (!provider) {
    return (
      <div className="h-dvh flex flex-col max-w-2xl mx-auto">
        <ProviderSelector onSelect={handleProviderSelect} />
      </div>
    );
  }

  // Screen 2: Storkreds selection
  if (!selectedStorkreds) {
    return (
      <div className="h-dvh flex flex-col max-w-2xl mx-auto">
        <StorkredsSelector
          storkredse={storkredse}
          onSelect={setSelectedStorkreds}
        />
      </div>
    );
  }

  // Screen 3: Chat / Results
  const chatReady = modelsReady && !!systemPrompt;

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="h-dvh flex flex-col max-w-2xl mx-auto">
        <header className="px-4 py-3 border-b">
          <div className="flex items-baseline gap-3">
            <h1 className="text-xl font-bold text-red-700">Kandidattest</h1>
            <span className="text-sm text-gray-500">{selectedStorkreds}</span>
            <span className="text-xs text-gray-400 ml-auto">
              {provider.label}: {provider.model}
            </span>
          </div>
          <nav className="flex gap-4 mt-2">
            <button
              onClick={() => setScreen("chat")}
              className={`text-sm pb-1 border-b-2 transition ${
                screen === "chat"
                  ? "border-red-600 text-red-700 font-medium"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              Samtale
            </button>
            <button
              onClick={() => setScreen("results")}
              className={`text-sm pb-1 border-b-2 transition ${
                screen === "results"
                  ? "border-red-600 text-red-700 font-medium"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              }`}
            >
              Resultater
            </button>
          </nav>
        </header>
        {screen === "chat" ? (
          <div className="flex-1 overflow-hidden">
            <ChatThread />
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <ResultsPage
              embeddingsRef={embeddingsRef}
              modelsReady={modelsReady}
              selectedStorkreds={selectedStorkreds!}
              candidates={candidates}
              provider={provider!}
              questions={questions}
              getUserMessages={getUserMessagesRef.current}
            />
          </div>
        )}
      </div>
      <ChoiceToolUI />
      <AutoStart ready={chatReady} />
      <UserMessageCollector messagesRef={getUserMessagesRef} />
    </AssistantRuntimeProvider>
  );
}
