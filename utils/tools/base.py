from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base class for tools usable by the LLM."""

    name: str
    description: str

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for the tool's parameters (OpenAI-compatible)."""
        pass

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """Execute the tool with the given arguments."""
        pass

    @property
    def openai_dict(self) -> Dict[str, Any]:
        """Convert the tool into OpenAI-compatible spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
