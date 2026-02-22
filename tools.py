import math
import datetime
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Tool functions
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


# ─────────────────────────────────────────────────────────────────────────────
# Registry & schemas
# ─────────────────────────────────────────────────────────────────────────────

TOOL_FUNCTIONS: dict[str, Any] = {
    "calculator":           calculator,
    "get_current_datetime": get_current_datetime,
    "read_file":            read_file,
    "write_file":           write_file,
    "web_search":           web_search,
    "get_weather":          get_weather,
}

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
