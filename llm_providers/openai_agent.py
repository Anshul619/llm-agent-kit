import os
import json
from openai import OpenAI

from llm_providers.base import BaseAgent, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, dispatch


class OpenAIAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["parameters"],
                },
            }
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
