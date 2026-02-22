"""
llm-agent-kit
A lightweight, provider-agnostic AI agent with tool calling support.

Usage:
  python main.py                                    # uses AGENT_PROVIDER env var (default: gemini)
  python main.py anthropic                          # Claude with default model
  python main.py openai gpt-4o                      # specific model
  python main.py gemini gemini-2.0-flash            # Gemini
  python main.py anthropic claude-3-5-haiku-20241022 "What is 2**32?"  # single query
"""

import os
import sys
from dotenv import load_dotenv

from llm_providers import BaseAgent, AnthropicAgent, GeminiAgent, OpenAIAgent

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER_DEFAULTS = {
    "anthropic": "claude-3-5-haiku-20241022",
    "gemini":    "gemini-2.0-flash",
    "openai":    "gpt-4o-mini",
}


def create_agent(provider: str, model: str | None = None) -> BaseAgent:
    provider = provider.lower()
    model = model or PROVIDER_DEFAULTS.get(provider)
    if provider == "anthropic":
        return AnthropicAgent(model)
    elif provider == "gemini":
        return GeminiAgent(model)
    elif provider == "openai":
        return OpenAIAgent(model)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose: anthropic, gemini, openai")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

def chat_loop(agent: BaseAgent):
    print("=" * 60)
    print("  llm-agent-kit  |  'quit' to stop  |  'reset' to clear history")
    print("=" * 60 + "\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("🔄  Conversation reset.\n")
            continue
        agent.run(user_input)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    provider = args[0] if len(args) >= 1 else os.environ.get("AGENT_PROVIDER", "gemini")
    model    = args[1] if len(args) >= 2 else None
    query    = " ".join(args[2:]) if len(args) >= 3 else None

    agent = create_agent(provider, model)

    if query:
        agent.run(query)
    else:
        chat_loop(agent)
