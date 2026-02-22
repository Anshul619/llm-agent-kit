from abc import ABC, abstractmethod

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools. "
    "Use tools whenever they help you give accurate answers. "
    "Think step-by-step for complex problems."
)


class BaseAgent(ABC):
    @abstractmethod
    def run(self, user_message: str) -> str: ...

    @abstractmethod
    def reset(self): ...
