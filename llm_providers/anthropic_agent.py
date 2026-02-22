import os
import anthropic

from llm_providers.base import BaseAgent, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, dispatch


class AnthropicAgent(BaseAgent):
    def __init__(self, model: str = "claude-3-5-haiku-20241022"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.history: list[dict] = []
        self.tools = [
            {
                "name": s["name"],
                "description": s["description"],
                "input_schema": s["parameters"],
            }
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

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses or response.stop_reason == "end_turn":
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                self.history.append({"role": "assistant", "content": response.content})
                print(f"🤖  Assistant: {text}\n")
                return text

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
