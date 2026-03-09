# Results Page with Candidate Profiles

## Date: 2026-03-09

## Summary

Replace the auto-triggered match flow with a freely navigable two-screen model (Chat / Results). Users switch to Results whenever they want to see embedding matches based on their chat so far. Each candidate expands accordion-style to show a profile card and an LLM-generated comparison.

## Navigation Flow

- Two screens: **Chat** and **Results**, switchable via top nav tabs
- No more `[KLAR TIL MATCH]` trigger — user decides when to check results
- Navigating to Results runs embedding match on-demand using all user messages collected so far
- Returning to Chat and adding more messages, then switching back to Results gives updated matches
- If embedding model isn't ready yet, show a loading state on the Results page

## Results Page

- Top 10 candidates ranked by cosine similarity (same as current `findMatches`)
- Each row: rank badge (party color), name, party letter, match %, occupation, area
- Clicking a row expands accordion-style below it

## Expanded Candidate View (Accordion)

### Profile Section
- Age, occupation, pitch (candidate intro)
- Priorities (5 key issues)
- Party name

### LLM Comparison Section
- Uses the same provider/model as the chat
- Prompt includes: user's chat messages + candidate's `answers` (score + comment per question) + `priorities`
- LLM identifies 3-5 most relevant points of agreement/disagreement
- Output: short Danish prose paragraphs
- Loading spinner while generating
- Cached per candidate ID (re-expanding doesn't re-generate, but new match run clears cache)

## Components

### New
- **ResultsPage** — container for the results list, triggers embedding on mount/navigation
- **CandidateCard** — expanded accordion content (profile + comparison)

### Modified
- **App.tsx** — add tab navigation state (chat/results), remove MatchDetector, remove debug button
- **CandidateResults.tsx** — refactor into ResultsPage with accordion behavior

### Removed
- `MatchDetector` component from App.tsx
- `[KLAR TIL MATCH]` instruction from system prompt
- Debug "Test match" button

## LLM Comparison Prompt

```
Du er en neutral politisk analytiker. Sammenlign brugerens holdninger med kandidatens svar.

Brugerens udtalelser:
{user messages joined}

Kandidat: {name} ({party})
Kandidatens svar på politiske spørgsmål:
{for each answer: question text, score (-2 to +2), comment}

Kandidatens prioriteter: {priorities}

Skriv 3-5 korte punkter på dansk om de vigtigste enigheder og uenigheder.
Hold det kort og neutralt.
```

## Data Flow

1. User chats on Chat screen (any number of messages)
2. User clicks "Resultater" tab
3. App collects all user messages from thread runtime
4. Embeds concatenated user text via EmbeddingManager
5. Runs findMatches → top 10 candidates
6. Renders results list
7. User clicks candidate → accordion expands
8. If no cached comparison: sends LLM comparison prompt via AI SDK → streams result
9. Caches result for that candidate
10. User can collapse, expand others, or go back to Chat
