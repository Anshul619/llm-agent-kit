"""
Multi-Provider AI Agent
Supports: Anthropic (Claude), Google (Gemini), OpenAI (GPT)
Switch providers via env var or CLI argument.
"""

import os
import sys
import math
import datetime
import json
from typing import Any
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
# ─────────────────────────────────────────────────────────────────────────────
# 1.  Tools (shared across all providers)
# ─────────────────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        return f"Result: {eval(expression, {'__builtins__': {}}, allowed)}"
    except Exception as e:
        return f"Error: {e}"


def get_current_datetime() -> str:
    return f"Current UTC: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"


def read_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(filepath: str, content: str) -> str:
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to '{filepath}'."
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str) -> str:
    """Simulated — swap in Tavily/SerpAPI for real results."""
    mock = {
        "anthropic": "Anthropic is an AI safety company that created Claude.",
        "openai":    "OpenAI is the company behind GPT-4 and ChatGPT.",
        "gemini":    "Gemini is Google DeepMind's multimodal AI model family.",
        "python":    "Python is a high-level, interpreted programming language.",
        "ai agent":  "An AI agent autonomously perceives its environment and takes actions to achieve goals.",
    }
    for k, v in mock.items():
        if k in query.lower():
            return v
    return f"No mock result for '{query}'. Integrate Tavily/SerpAPI for live search."


def get_weather(city: str) -> str:
    """Simulated — swap in OpenWeatherMap for real data."""
    mock = {
        "london":   "London: 15°C, overcast, humidity 78%.",
        "new york": "New York: 22°C, partly cloudy, humidity 55%.",
        "tokyo":    "Tokyo: 28°C, sunny, humidity 65%.",
        "sydney":   "Sydney: 19°C, light rain, humidity 80%.",
    }
    return mock.get(city.lower(), f"No weather data for '{city}' in simulation.")


# Tool registry
TOOL_FUNCTIONS: dict[str, Any] = {
    "calculator":           calculator,
    "get_current_datetime": get_current_datetime,
    "read_file":            read_file,
    "write_file":           write_file,
    "web_search":           web_search,
    "get_weather":          get_weather,
}

# Shared tool schema (converted per-provider below)
TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression (supports sqrt, sin, cos, log, **, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. 'sqrt(144) + 2**8'"}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Returns the current UTC date and time.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read a local text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the file."}
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text to a local file (creates or overwrites).",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string"},
                "content":  {"type": "string"},
            },
            "required": ["filepath", "content"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo'."}
            },
            "required": ["city"],
        },
    },
]


def dispatch(name: str, args: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Unknown tool: '{name}'"
    try:
        return func(**args)
    except Exception as e:
        return f"Tool error: {e}"


SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. "
    "Use tools whenever they help you give accurate answers. "
    "Think step-by-step for complex problems."
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Provider backends
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    @abstractmethod
    def run(self, user_message: str) -> str: ...
    @abstractmethod
    def reset(self): ...


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicAgent(BaseAgent):
    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.history: list[dict] = []

        # Anthropic tool format
        self.tools = [
            {"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
            for s in TOOL_SCHEMAS
        ]
        print(f"✅  Anthropic Agent ready  ({model})\n")

    def run(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        print(f"👤  User: {user_message}")

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=self.history,
            )

            # Collect tool use blocks
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses or response.stop_reason == "end_turn":
                # Final text response
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                self.history.append({"role": "assistant", "content": response.content})
                print(f"🤖  Assistant: {text}\n")
                return text

            # Add assistant message and tool results
            self.history.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                print(f"  🔧  [{self.model}] Calling '{tu.name}' with {dict(tu.input)}")
                result = dispatch(tu.name, dict(tu.input))
                print(f"  📤  Result: {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })
            self.history.append({"role": "user", "content": tool_results})

    def reset(self):
        self.history = []


# ── Google Gemini ─────────────────────────────────────────────────────────────
class GeminiAgent(BaseAgent):
    def __init__(self, model: str = "gemini-2.0-flash"):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")

        self.genai = genai
        self.types = types
        self.model_name = model
        self.client = genai.Client(api_key=api_key)

        self.tools = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=s["name"],
                    description=s["description"],
                    parameters=s["parameters"],
                )
                for s in TOOL_SCHEMAS
            ]
        )

        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[self.tools],
        )

        self._new_chat()
        print(f"✅  Gemini Agent ready  ({model})\n")

    def _new_chat(self):
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=self.config,
        )

    def run(self, user_message: str) -> str:
        print(f"👤  User: {user_message}")
        response = self.chat.send_message(user_message)

        while True:
            calls = [
                part.function_call
                for candidate in response.candidates
                for part in candidate.content.parts
                if part.function_call is not None and part.function_call.name
            ]

            if not calls:
                break

            parts = []
            for fc in calls:
                args = dict(fc.args)
                print(f"  🔧  [{self.model_name}] Calling '{fc.name}' with {args}")
                result = dispatch(fc.name, args)
                print(f"  📤  Result: {result}")
                parts.append(
                    self.types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )

            response = self.chat.send_message(parts)

        text = "".join(
            part.text
            for c in response.candidates
            for part in c.content.parts
            if hasattr(part, "text") and part.text
        ).strip()
        print(f"🤖  Assistant: {text}\n")
        return text

    def reset(self):
        self._new_chat()

# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # OpenAI tool format
        self.tools = [
            {"type": "function", "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            }}
            for s in TOOL_SCHEMAS
        ]
        print(f"✅  OpenAI Agent ready  ({model})\n")

    def run(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        print(f"👤  User: {user_message}")

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=self.tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                text = msg.content or ""
                self.history.append({"role": "assistant", "content": text})
                print(f"🤖  Assistant: {text}\n")
                return text

            self.history.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"  🔧  [{self.model}] Calling '{tc.function.name}' with {args}")
                result = dispatch(tc.function.name, args)
                print(f"  📤  Result: {result}")
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Factory
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER_DEFAULTS = {
    "anthropic": "claude-3-5-haiku-20241022",
    "gemini":    "gemini-2.5-flash",
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
# 4.  Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

def chat_loop(agent: BaseAgent):
    print("=" * 60)
    print("  Multi-Provider AI Agent  |  'quit' to stop  |  'reset' to clear history")
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
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage:
      python agent.py                                   # uses AGENT_PROVIDER env var (default: gemini)
      python agent.py anthropic                         # Claude with default model
      python agent.py openai gpt-4o                    # specific model
      python agent.py gemini gemini-2.0-flash-exp      # Gemini experimental
      python agent.py anthropic claude-opus-4-5 "What is 2**32?"  # single query
    """
    args = sys.argv[1:]

    provider = args[0] if len(args) >= 1 else os.environ.get("AGENT_PROVIDER", "gemini")
    model    = args[1] if len(args) >= 2 else None
    query    = " ".join(args[2:]) if len(args) >= 3 else None

    agent = create_agent(provider, model)

    if query:
        agent.run(query)
    else:
        chat_loop(agent)
