import { makeAssistantToolUI, useThreadRuntime } from "@assistant-ui/react";

/**
 * Renders [VALG: ...] choices as clickable buttons.
 * When clicked, sends the chosen option as a new user message.
 */
export const ChoiceToolUI = makeAssistantToolUI<
  { options: string[] },
  string
>({
  toolName: "user_choice",
  render: ({ args, result, addResult, status }) => {
    const threadRuntime = useThreadRuntime();
    const isActive =
      status.type === "requires-action" && !result;

    const handleClick = (option: string) => {
      if (!isActive) return;
      // Provide the tool result so the message completes
      addResult(option);
      // Also append as a user message so the LLM sees it in conversation
      threadRuntime.append({
        role: "user",
        content: [{ type: "text", text: option }],
      });
    };

    if (result) {
      return (
        <div className="mt-2">
          <span className="inline-block bg-ft-red-light text-ft-red-dark text-sm px-3 py-1 rounded-full border border-ft-red/20">
            {result}
          </span>
        </div>
      );
    }

    return (
      <div className="flex flex-wrap gap-2 mt-3">
        {args.options.map((option) => (
          <button
            key={option}
            onClick={() => handleClick(option)}
            disabled={!isActive}
            className="px-4 py-2 bg-white border border-gray-300 rounded-full text-sm font-medium text-gray-700 hover:bg-ft-red-light hover:border-ft-red/30 hover:text-ft-red-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {option}
          </button>
        ))}
      </div>
    );
  },
});
