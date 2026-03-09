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
import { createLLMAdapter } from "./adapters/chat-model-adapter";
import { createApiAdapter } from "./adapters/api-adapter";
import { EmbeddingManager } from "./lib/embeddings";
import type { CandidateMatch } from "./lib/embeddings";
import { buildLocalPrompt, buildApiPrompt } from "./lib/system-prompt";
import { checkWebGPU } from "./lib/webgpu";
import type { Provider } from "./lib/provider";
import { ProviderSelector } from "./components/ProviderSelector";
import { StorkredsSelector } from "./components/StorkredsSelector";
import { CandidateResults } from "./components/CandidateResults";
import { ChoiceToolUI } from "./components/ChoiceButtons";

interface Storkreds {
  name: string;
  candidateCount: number;
}

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

function ModelLoadingOverlay({ items }: { items: Map<string, number> }) {
  if (items.size === 0) return null;
  return (
    <div className="fixed inset-0 bg-black/60 flex flex-col items-center justify-center z-50">
      <h2 className="text-white text-xl font-bold mb-6">
        Henter AI-modeller...
      </h2>
      <p className="text-white/70 text-sm mb-4">
        Første gang tager det et par minutter. Modellerne caches i browseren.
      </p>
      {[...items.entries()].map(([file, progress]) => (
        <div key={file} className="w-80 mb-3">
          <p className="text-white/80 text-xs truncate">{file}</p>
          <div className="bg-white/20 rounded-full h-2 mt-1">
            <div
              className="bg-red-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// --- Thread UI built from assistant-ui primitives ---

function TextPart() {
  const part = useMessagePartText();
  const isThinking = part.text.includes("💭 Tænker...");

  if (isThinking) {
    const visibleText = part.text.replace(/\n*💭 Tænker\.\.\.$/, "").trim();
    return (
      <>
        {visibleText && <span>{visibleText}</span>}
        <span className="block text-gray-400 italic text-sm animate-pulse mt-1">
          Tænker...
        </span>
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

// --- Match detection ---

function MatchDetector({
  onMatch,
  matchTriggered,
}: {
  onMatch: (userMessages: string[]) => void;
  matchTriggered: boolean;
}) {
  const threadRuntime = useThreadRuntime();

  useEffect(() => {
    if (matchTriggered) return;

    const unsubscribe = threadRuntime.subscribe(() => {
      const state = threadRuntime.getState();
      const messages = state.messages;
      if (!messages?.length) return;

      const lastMsg = messages[messages.length - 1];
      if (lastMsg.role !== "assistant") return;

      const text = lastMsg.content
        ?.filter(
          (p): p is { type: "text"; text: string } => p.type === "text",
        )
        .map((p) => p.text)
        .join("");

      if (text?.includes("[KLAR TIL MATCH]")) {
        const userMessages = messages
          .filter((m) => m.role === "user")
          .map((m) =>
            m.content
              .filter(
                (p): p is { type: "text"; text: string } => p.type === "text",
              )
              .map((p) => p.text)
              .join(""),
          );
        onMatch(userMessages);
      }
    });

    return unsubscribe;
  }, [threadRuntime, onMatch, matchTriggered]);

  return null;
}

// --- Main App ---

export default function App() {
  const llmWorkerRef = useRef<Worker | null>(null);
  const embeddingsRef = useRef<EmbeddingManager | null>(null);
  const [provider, setProvider] = useState<Provider | null>(null);
  const [progressItems, setProgressItems] = useState<Map<string, number>>(
    new Map(),
  );
  const [modelsReady, setModelsReady] = useState(false);
  const [selectedStorkreds, setSelectedStorkreds] = useState<string | null>(
    null,
  );
  const [storkredse, setStorkredse] = useState<Storkreds[]>([]);
  const [webgpuSupported, setWebgpuSupported] = useState(true);
  const [matches, setMatches] = useState<CandidateMatch[] | null>(null);
  const [candidates, setCandidates] = useState<Map<number, CandidateData>>(
    new Map(),
  );
  const [systemPrompt, setSystemPrompt] = useState<string | null>(null);

  const handleProgress = useCallback(
    (data: { status?: string; file?: string; progress?: number }) => {
      if (!data.file) return;
      if (data.status === "initiate" || data.status === "download") {
        setProgressItems((prev) => new Map(prev).set(data.file!, 0));
      } else if (data.status === "progress") {
        setProgressItems(
          (prev) => new Map(prev).set(data.file!, data.progress ?? 0),
        );
      } else if (data.status === "done") {
        setProgressItems((prev) => {
          const next = new Map(prev);
          next.delete(data.file!);
          return next;
        });
      }
    },
    [],
  );

  // Check WebGPU + load static data on mount
  useEffect(() => {
    checkWebGPU().then((result) => {
      if (!result.supported) {
        setWebgpuSupported(false);
      }
    });

    fetch("/storkredse.json")
      .then((r) => r.json())
      .then(setStorkredse);

    fetch("/candidates.json")
      .then((r) => r.json())
      .then((data: CandidateData[]) => {
        setCandidates(new Map(data.map((c) => [c.id, c])));
      });
  }, []);

  // Load models when provider is selected
  useEffect(() => {
    if (!provider) return;

    async function loadModels() {
      // Always load embeddings
      try {
        embeddingsRef.current = new EmbeddingManager(handleProgress);
        await embeddingsRef.current.load();
        console.log("[APP] Embedding model loaded");
      } catch (e) {
        console.error("[APP] Embedding model failed:", e);
      }

      // Only load LLM worker for local provider
      if (provider!.type === "local") {
        try {
          const worker = new Worker(
            new URL("./workers/llm-worker.ts", import.meta.url),
            { type: "module" },
          );
          llmWorkerRef.current = worker;

          worker.addEventListener("message", (e) => handleProgress(e.data));

          await new Promise<void>((resolve, reject) => {
            const handler = (e: MessageEvent) => {
              if (e.data.type === "ready") {
                worker.removeEventListener("message", handler);
                resolve();
              }
            };
            worker.addEventListener("message", handler);
            worker.addEventListener("error", (e) => {
              console.error("[APP] LLM worker error:", e);
              reject(e);
            });
            worker.postMessage({ type: "load" });
          });
          console.log("[APP] LLM model loaded");
        } catch (e) {
          console.error("[APP] LLM model failed:", e);
        }
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

      const promptBuilder =
        provider!.type === "api" ? buildApiPrompt : buildLocalPrompt;
      setSystemPrompt(
        promptBuilder(national, localForStorkreds, selectedStorkreds!),
      );
    }

    loadPrompt();
  }, [selectedStorkreds, provider]);

  // ProviderSelector handles sessionStorage internally
  const handleProviderSelect = useCallback((p: Provider) => {
    setProvider(p);
  }, []);

  // Build adapter based on provider type
  const adapter =
    provider?.type === "api"
      ? createApiAdapter(
          provider.baseUrl,
          provider.apiKey,
          provider.model,
          systemPrompt ?? "",
        )
      : createLLMAdapter(llmWorkerRef.current!, systemPrompt ?? "");

  const runtime = useLocalRuntime(adapter);

  const handleMatchDetected = useCallback(
    async (userMessages: string[]) => {
      if (!embeddingsRef.current || !modelsReady) return;
      const userText = userMessages.join("\n\n");
      const embedding = await embeddingsRef.current.embed(userText);
      const results = embeddingsRef.current.findMatches(
        embedding,
        10,
        selectedStorkreds ?? undefined,
      );
      setMatches(results);
    },
    [modelsReady, selectedStorkreds],
  );

  // --- Render screens ---

  // Screen 1: Provider selection
  if (!provider) {
    return (
      <div className="h-dvh flex flex-col max-w-2xl mx-auto">
        <ProviderSelector
          webgpuSupported={webgpuSupported}
          onSelect={handleProviderSelect}
        />
      </div>
    );
  }

  // Screen 2: Storkreds selection
  if (!selectedStorkreds) {
    return (
      <>
        {provider.type === "local" && (
          <ModelLoadingOverlay items={progressItems} />
        )}
        <div className="h-dvh flex flex-col max-w-2xl mx-auto">
          <StorkredsSelector
            storkredse={storkredse}
            onSelect={setSelectedStorkreds}
          />
        </div>
      </>
    );
  }

  // Screen 3: Chat
  const chatReady = modelsReady && !!systemPrompt;

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {provider.type === "local" && (
        <ModelLoadingOverlay items={progressItems} />
      )}
      <div className="h-dvh flex flex-col max-w-2xl mx-auto">
        <header className="px-4 py-3 border-b flex items-baseline gap-3">
          <h1 className="text-xl font-bold text-red-700">Kandidattest</h1>
          <span className="text-sm text-gray-500">{selectedStorkreds}</span>
          <span className="text-xs text-gray-400 ml-auto">
            {provider.type === "api" ? provider.model : "Qwen3 1.7B (lokal)"}
          </span>
        </header>
        <div className="flex-1 overflow-hidden">
          <ChatThread />
        </div>
        {matches && (
          <div className="border-t max-h-[50vh] overflow-y-auto">
            <CandidateResults matches={matches} candidates={candidates} />
          </div>
        )}
      </div>
      <ChoiceToolUI />
      <AutoStart ready={chatReady} />
      <MatchDetector
        onMatch={handleMatchDetected}
        matchTriggered={matches !== null}
      />
    </AssistantRuntimeProvider>
  );
}
