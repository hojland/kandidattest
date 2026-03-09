# Results Page Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace auto-triggered match with a two-tab Chat/Results UI where users freely switch between screens, candidates expand accordion-style with LLM-generated comparisons.

**Architecture:** App.tsx gains a `screen` state ("chat" | "results"). Tab nav in the header switches screens. Results page collects user messages from thread runtime, embeds them, runs findMatches. Clicking a candidate expands an accordion showing profile + streamed LLM comparison. CandidateData interface is expanded to include answers/priorities from candidates.json.

**Tech Stack:** React, assistant-ui thread runtime, Vercel AI SDK `streamText`, Tailwind CSS

---

### Task 1: Expand CandidateData and load full candidate data

**Files:**
- Modify: `src/App.tsx:26-35` (CandidateData interface)

**Step 1: Add answers and priorities to CandidateData interface**

In `src/App.tsx`, replace the CandidateData interface:

```typescript
interface CandidateAnswer {
  score: number;
  comment: string;
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
  priorities?: string[];
  answers?: Record<string, CandidateAnswer>;
}
```

No other changes needed — `candidates.json` already contains these fields, they just weren't typed. The existing `fetch("/candidates.json")` call already loads them.

**Step 2: Verify build**

Run: `npm run build`
Expected: success (no new errors)

**Step 3: Commit**

```bash
git add src/App.tsx
git commit -m "feat: expand CandidateData to include answers and priorities"
```

---

### Task 2: Remove MatchDetector and debug code, add tab navigation state

**Files:**
- Modify: `src/App.tsx`

**Step 1: Remove MatchDetector, debug button, handleTestMatch, handleMatchDetected**

Delete from App.tsx:
- The `MatchDetector` component (lines 166-214)
- `handleMatchDetected` callback (lines 308-324)
- `handleTestMatch` callback (lines 326-343)
- The `<MatchDetector>` JSX element in the render (lines 398-401)
- The "Test match" button in the header (lines 377-382)
- The matches panel in the chat view (lines 390-394)
- The `matches` state — move it to be set by the results page instead

**Step 2: Add screen state and tab navigation**

Add state to App:
```typescript
const [screen, setScreen] = useState<"chat" | "results">("chat");
```

Replace the header with tab navigation:
```tsx
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
```

**Step 3: Conditionally render chat or results**

Replace the chat body area with:
```tsx
{screen === "chat" ? (
  <div className="flex-1 overflow-hidden">
    <ChatThread />
  </div>
) : (
  <div className="flex-1 overflow-y-auto">
    <ResultsPage
      embeddingsRef={embeddingsRef}
      modelsReady={modelsReady}
      selectedStorkreds={selectedStorkreds}
      candidates={candidates}
      provider={provider}
      questions={questions}
    />
  </div>
)}
```

Add a `questions` state to store the loaded question map (needed for comparison prompt):
```typescript
const [questions, setQuestions] = useState<Record<string, string>>({});
```

Load it alongside the system prompt (in the existing `loadPrompt` effect):
```typescript
setQuestions(national);
```

Add a placeholder `ResultsPage` import at the top (we'll create the component next).

**Step 4: Verify build**

Run: `npm run build`
Expected: success

**Step 5: Commit**

```bash
git add src/App.tsx
git commit -m "feat: add chat/results tab navigation, remove MatchDetector"
```

---

### Task 3: Create ResultsPage component

**Files:**
- Create: `src/components/ResultsPage.tsx`

**Step 1: Create the component**

```tsx
import { useState, useEffect, useRef, useCallback } from "react";
import type { EmbeddingManager, CandidateMatch } from "../lib/embeddings";
import type { Provider } from "../lib/provider";
import { CandidateCard } from "./CandidateCard";

interface CandidateAnswer {
  score: number;
  comment: string;
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
  priorities?: string[];
  answers?: Record<string, CandidateAnswer>;
}

interface Props {
  embeddingsRef: React.RefObject<EmbeddingManager | null>;
  modelsReady: boolean;
  selectedStorkreds: string;
  candidates: Map<number, CandidateData>;
  provider: Provider;
  questions: Record<string, string>;
  getUserMessages: () => string[];
}

const PARTY_COLORS: Record<string, string> = {
  a: "#A82721", b: "#733280", c: "#00583C", d: "#00505C",
  f: "#E4007F", i: "#00B0CA", k: "#F29D00", m: "#532380",
  o: "#005078", v: "#004A98", ø: "#E5001A", å: "#2B8738",
};

export function ResultsPage({
  embeddingsRef,
  modelsReady,
  selectedStorkreds,
  candidates,
  provider,
  questions,
  getUserMessages,
}: Props) {
  const [matches, setMatches] = useState<CandidateMatch[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const comparisonCache = useRef<Map<number, string>>(new Map());

  const runMatch = useCallback(async () => {
    if (!embeddingsRef.current || !modelsReady) {
      setError("Embedding-modellen indlæses stadig...");
      return;
    }

    const userMessages = getUserMessages();
    if (userMessages.length === 0) {
      setError("Skriv noget i chatten først, så vi kan matche dig.");
      return;
    }

    setLoading(true);
    setError(null);
    comparisonCache.current.clear();

    try {
      const userText = userMessages.join("\n\n");
      const embedding = await embeddingsRef.current.embed(userText);
      const hasNaN = embedding.some((v: number) => isNaN(v));
      if (hasNaN) {
        setError("Embedding-modellen returnerede ugyldige data. Prøv igen.");
        setLoading(false);
        return;
      }
      const results = embeddingsRef.current.findMatches(
        embedding, 10, selectedStorkreds ?? undefined,
      );
      setMatches(results);
    } catch (e) {
      setError("Kunne ikke beregne match. Prøv igen.");
      console.error("[RESULTS] Match error:", e);
    } finally {
      setLoading(false);
    }
  }, [embeddingsRef, modelsReady, selectedStorkreds, getUserMessages]);

  // Run match on mount
  useEffect(() => { runMatch(); }, [runMatch]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full p-8">
        <p className="text-gray-500 animate-pulse">Beregner matches...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 gap-4">
        <p className="text-gray-500">{error}</p>
        <button
          onClick={runMatch}
          className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700"
        >
          Prøv igen
        </button>
      </div>
    );
  }

  if (!matches || matches.length === 0) {
    return (
      <div className="flex items-center justify-center h-full p-8">
        <p className="text-gray-500">Ingen matches fundet.</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-2">
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="text-lg font-bold text-gray-900">Dine top-matches</h2>
        <button
          onClick={runMatch}
          className="text-xs text-red-600 hover:text-red-700 font-medium"
        >
          Opdater matches
        </button>
      </div>
      {matches.map((match, i) => {
        const c = candidates.get(match.id);
        const color = PARTY_COLORS[c?.partyLetter?.toLowerCase() ?? ""] || "#666";
        const isExpanded = expandedId === match.id;

        return (
          <div key={match.id} className="rounded-lg border bg-white shadow-sm overflow-hidden">
            <button
              onClick={() => setExpandedId(isExpanded ? null : match.id)}
              className="flex items-start gap-3 p-3 w-full text-left hover:bg-gray-50 transition"
            >
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm shrink-0"
                style={{ backgroundColor: color }}
              >
                {i + 1}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-2">
                  <span className="font-semibold text-gray-900">{match.name}</span>
                  <span className="text-sm text-gray-500">({c?.partyLetter?.toUpperCase()})</span>
                  <span className="ml-auto text-sm font-medium text-red-700">
                    {Math.round(match.score * 100)}%
                  </span>
                </div>
                {c?.occupation && <p className="text-sm text-gray-500">{c.occupation}</p>}
              </div>
              <span className="text-gray-400 text-sm mt-1">{isExpanded ? "▾" : "▸"}</span>
            </button>
            {isExpanded && c && (
              <CandidateCard
                candidate={c}
                questions={questions}
                provider={provider}
                userMessages={getUserMessages()}
                comparisonCache={comparisonCache}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
```

**Step 2: Verify build (will fail — CandidateCard doesn't exist yet)**

That's expected. Move to next task.

---

### Task 4: Create CandidateCard component with profile and LLM comparison

**Files:**
- Create: `src/components/CandidateCard.tsx`

**Step 1: Create the component**

```tsx
import { useState, useEffect, useCallback } from "react";
import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import type { Provider } from "../lib/provider";

interface CandidateAnswer {
  score: number;
  comment: string;
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
  priorities?: string[];
  answers?: Record<string, CandidateAnswer>;
}

interface Props {
  candidate: CandidateData;
  questions: Record<string, string>;
  provider: Provider;
  userMessages: string[];
  comparisonCache: React.RefObject<Map<number, string>>;
}

const SCORE_LABELS: Record<number, string> = {
  "-2": "Helt uenig",
  "-1": "Delvist uenig",
  "0": "Neutral",
  "1": "Delvist enig",
  "2": "Helt enig",
};

function createModel(provider: Provider) {
  switch (provider.kind) {
    case "anthropic":
      return createAnthropic({ apiKey: provider.apiKey })(provider.model);
    case "google":
      return createGoogleGenerativeAI({ apiKey: provider.apiKey })(provider.model);
    case "openai":
      return createOpenAI({ apiKey: provider.apiKey })(provider.model);
    case "openai-compatible":
      return createOpenAI({ apiKey: provider.apiKey, baseURL: provider.baseUrl })(provider.model);
  }
}

function buildComparisonPrompt(
  candidate: CandidateData,
  questions: Record<string, string>,
  userMessages: string[],
): string {
  const userText = userMessages.join("\n\n");

  const answerLines = Object.entries(candidate.answers ?? {})
    .filter(([key]) => questions[key])
    .map(([key, ans]) => {
      const label = SCORE_LABELS[ans.score] ?? `${ans.score}`;
      const comment = ans.comment ? ` — "${ans.comment}"` : "";
      return `- ${questions[key]}: ${label}${comment}`;
    })
    .join("\n");

  return `Du er en neutral politisk analytiker. Sammenlign brugerens holdninger med denne kandidats svar og prioriteter.

Brugerens udtalelser fra samtalen:
${userText}

Kandidat: ${candidate.name} (${candidate.party})
${candidate.age ? `Alder: ${candidate.age}` : ""}
${candidate.occupation ? `Stilling: ${candidate.occupation}` : ""}

Kandidatens prioriteter: ${(candidate.priorities ?? []).join(", ")}

Kandidatens svar på politiske spørgsmål:
${answerLines}

Skriv 3-5 korte punkter på dansk om de vigtigste enigheder og uenigheder mellem brugeren og kandidaten. Fokuser på det mest relevante baseret på hvad brugeren faktisk har udtalt sig om. Hold det kort, neutralt og konkret.`;
}

export function CandidateCard({
  candidate,
  questions,
  provider,
  userMessages,
  comparisonCache,
}: Props) {
  const [comparison, setComparison] = useState<string>(
    () => comparisonCache.current.get(candidate.id) ?? "",
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateComparison = useCallback(async () => {
    if (comparison) return; // Already cached
    if (userMessages.length === 0) {
      setError("Ingen brugerdata at sammenligne med.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const model = createModel(provider);
      const prompt = buildComparisonPrompt(candidate, questions, userMessages);

      const result = streamText({
        model,
        prompt,
        temperature: 0.5,
        maxOutputTokens: 1024,
      });

      let text = "";
      for await (const part of result.textStream) {
        text += part;
        setComparison(text);
      }

      comparisonCache.current.set(candidate.id, text);
    } catch (e) {
      console.error("[COMPARISON] Error:", e);
      setError("Kunne ikke generere sammenligning. Prøv igen.");
    } finally {
      setLoading(false);
    }
  }, [candidate, questions, provider, userMessages, comparison, comparisonCache]);

  // Auto-generate on mount
  useEffect(() => { generateComparison(); }, [generateComparison]);

  return (
    <div className="border-t px-4 py-3 space-y-3 bg-gray-50">
      {/* Profile section */}
      <div className="space-y-1">
        <p className="font-medium text-gray-900">
          {candidate.name}, {candidate.age ? `${candidate.age} år` : ""} — {candidate.party}
        </p>
        {candidate.occupation && (
          <p className="text-sm text-gray-600">{candidate.occupation}</p>
        )}
        {candidate.pitch && (
          <p className="text-sm text-gray-600 italic">{candidate.pitch}</p>
        )}
        {candidate.priorities && candidate.priorities.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {candidate.priorities.map((p) => (
              <span key={p} className="text-xs bg-gray-200 text-gray-700 px-2 py-0.5 rounded-full">
                {p}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Comparison section */}
      <div className="border-t pt-3">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Sammenligning med dine holdninger</h4>
        {error ? (
          <div className="text-sm text-red-600">
            <p>{error}</p>
            <button
              onClick={() => { setError(null); setComparison(""); generateComparison(); }}
              className="text-red-700 underline text-xs mt-1"
            >
              Prøv igen
            </button>
          </div>
        ) : comparison ? (
          <div className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
            {comparison}
          </div>
        ) : loading ? (
          <p className="text-sm text-gray-400 animate-pulse">Genererer sammenligning...</p>
        ) : null}
      </div>
    </div>
  );
}
```

**Step 2: Verify build**

Run: `npm run build`
Expected: success

**Step 3: Commit**

```bash
git add src/components/CandidateCard.tsx src/components/ResultsPage.tsx
git commit -m "feat: add ResultsPage and CandidateCard components"
```

---

### Task 5: Wire ResultsPage into App.tsx

**Files:**
- Modify: `src/App.tsx`

**Step 1: Add getUserMessages helper using thread runtime**

We need a way for ResultsPage to get user messages from the assistant-ui thread runtime. Create a `UserMessageCollector` component that exposes user messages via a ref:

```tsx
function UserMessageCollector({ messagesRef }: { messagesRef: React.MutableRefObject<() => string[]> }) {
  const threadRuntime = useThreadRuntime();

  messagesRef.current = () => {
    const state = threadRuntime.getState();
    return (state.messages ?? [])
      .filter((m) => m.role === "user")
      .map((m) =>
        m.content
          .filter((p): p is { type: "text"; text: string } => p.type === "text")
          .map((p) => p.text)
          .join(""),
      );
  };

  return null;
}
```

Add a ref in App:
```typescript
const getUserMessagesRef = useRef<() => string[]>(() => []);
```

Render `<UserMessageCollector messagesRef={getUserMessagesRef} />` inside the `<AssistantRuntimeProvider>`.

Pass `getUserMessages={getUserMessagesRef.current}` to `ResultsPage`.

**Step 2: Import ResultsPage, add questions state, wire everything**

Full changes to App.tsx:
- Add import: `import { ResultsPage } from "./components/ResultsPage";`
- Remove import: `import { CandidateResults } from "./components/CandidateResults";`
- Remove import: `import type { CandidateMatch } from "./lib/embeddings";`
- Add `questions` state: `const [questions, setQuestions] = useState<Record<string, string>>({});`
- In the `loadPrompt` effect, add `setQuestions(national);` after `setSystemPrompt(...)`
- Add `screen` state: `const [screen, setScreen] = useState<"chat" | "results">("chat");`
- Remove `matches` state entirely

**Step 3: Verify build**

Run: `npm run build`
Expected: success

**Step 4: Commit**

```bash
git add src/App.tsx
git commit -m "feat: wire ResultsPage into App with tab navigation"
```

---

### Task 6: Remove [KLAR TIL MATCH] from system prompt

**Files:**
- Modify: `src/lib/system-prompt.ts`

**Step 1: Update the system prompt**

In `buildApiPrompt`, modify TRIN 5 (lines ~217-219):

Replace:
```
TRIN 5 — AFSLUTNING
- Når du har dækket mindst 15 nationale + 3 lokale spørgsmål, opsummer kort hvad du har hørt
- Skriv derefter præcis dette på en ny linje: [KLAR TIL MATCH]
```

With:
```
TRIN 5 — AFSLUTNING
- Når du har dækket mindst 15 nationale + 3 lokale spørgsmål, opsummer kort hvad du har hørt
- Fortæl brugeren at de nu kan klikke på "Resultater"-fanen for at se deres matches
```

Also remove from VIGTIGE REGLER (line ~230):
```
- Skriv ALDRIG [KLAR TIL MATCH] før du har dækket nok spørgsmål
```

**Step 2: Verify build**

Run: `npm run build`
Expected: success

**Step 3: Commit**

```bash
git add src/lib/system-prompt.ts
git commit -m "feat: update system prompt to reference results tab instead of KLAR TIL MATCH"
```

---

### Task 7: Clean up dead code

**Files:**
- Delete or gut: `src/adapters/chat-model-adapter.ts` (no longer imported)
- Delete or gut: `src/adapters/api-adapter.ts` (no longer imported)
- Delete or gut: `src/lib/webgpu.ts` (no longer imported)
- Delete or gut: `src/workers/llm-worker.ts` (no longer used)
- Verify: `src/components/CandidateResults.tsx` is no longer imported anywhere

**Step 1: Remove dead files**

Check that nothing imports these files, then delete them.

**Step 2: Verify build**

Run: `npm run build`
Expected: success

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove dead code (local LLM adapter, WebGPU check, old results component)"
```

---

### Task 8: Manual test

**Step 1: Run dev server and test the full flow**

Run: `npm run dev`

1. Select a provider (Gemini or Anthropic)
2. Select a storkreds
3. Chat a few messages expressing political opinions
4. Click "Resultater" tab — verify matches appear with percentages
5. Click a candidate — verify accordion expands with profile and streaming comparison
6. Click "Samtale" tab — verify chat is intact
7. Chat more, click "Resultater" again — verify updated matches
8. Click "Opdater matches" button — verify re-runs

**Step 2: Commit any fixes**
