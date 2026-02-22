import os
from google import genai
from google.genai import types

from llm_providers.base import BaseAgent, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, dispatch


class GeminiAgent(BaseAgent):
    def __init__(self, model: str = "gemini-2.0-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model

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
                    types.Part.from_function_response(
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
