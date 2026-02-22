# 🤖 llm-agent-kit

A lightweight, provider-agnostic AI agent with tool calling support for Anthropic, Google Gemini, and OpenAI. Switch providers via CLI or environment variable.

| Provider | Models |
|---|---|
| **Anthropic** | claude-3-5-haiku, claude-3-5-sonnet, claude-opus-4-5, ... |
| **Google Gemini** | gemini-2.5-flash, gemini-2.5-pro, gemini-2.5-flash-exp, ... |
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-4-turbo, o1-mini, ... |

All providers share the same 6 tools: 
- 🧮 calculator
- 🕐 datetime
- 📂 read file
- 💾 write file
- 🔍 web search
- 🌤️ weather

## 🚀 Setup

### Create and activate virtual environment

````shell
python3 -m venv .venv
source .venv/bin/activate
````

### Install dependencies

Install all three providers, or just the one(s) you need:

```bash
pip install -r requirements.txt
```

### Set environment variables
- Copy .env.example to .env
- Set API key(s)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
export OPENAI_API_KEY="sk-..."
```

Get your keys:
- Anthropic → https://console.anthropic.com
- Gemini    → https://aistudio.google.com/app/apikey
- OpenAI    → https://platform.openai.com/api-keys

---

## ▶️ Running the Agent

### Interactive chat

```bash
python main.py                          # default: Gemini flash

python main.py anthropic                # Claude (default model)
python main.py openai                   # GPT-4o-mini (default)
python main.py gemini                   # Gemini 2.5 Flash (default)

# Pick any specific model
python main.py anthropic claude-3-5-sonnet-20241022
python main.py anthropic claude-opus-4-5
python main.py openai gpt-4o
python main.py openai o1-mini
python main.py gemini gemini-2.5-pro
python main.py gemini gemini-2.0-flash-001

# Single-shot query (no interactive mode)
python main.py anthropic claude-3-5-haiku-20241022 "What is sqrt(1764)?"
python main.py openai gpt-4o-mini "What time is it?"
python main.py gemini gemini-2.5-flash "Write a haiku and save it to haiku.txt"
```

## 💬 Example Session

```
$ python main.py openai gpt-4o-mini

✅  OpenAI Agent ready  (gpt-4o-mini)

============================================================
  Multi-Provider AI Agent  |  'quit' to stop  |  'reset' to clear history
============================================================

You: What is 10% of 4,500 plus the square root of 256?

You: reset
🔄  Conversation reset.

You: quit
Goodbye!
```

---

## 🗂️ Project Structure

```
llm-agent-kit/
├── main.py               # Entry point, factory, REPL
├── tools.py              # Tool functions and schemas (shared)
├── agents/
│   ├── __init__.py
│   ├── base.py           # BaseAgent ABC + SYSTEM_PROMPT
│   ├── anthropic_agent.py
│   ├── gemini_agent.py
│   └── openai_agent.py
├── requirements.txt
└── .env
```
