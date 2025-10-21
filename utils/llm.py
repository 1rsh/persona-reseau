import instructor
from openai import AsyncOpenAI
from abc import ABC, abstractmethod
from typing import List, Literal, Union
from pydantic import BaseModel
import json
import aiohttp

from utils.logger import logger
from utils.constants import PROVIDER_INFORMATION
from utils.tools.base import BaseTool

class BaseLLMService(ABC):
    @abstractmethod
    async def call_llm(self, model: str, messages: List[dict]):
        pass

    @abstractmethod
    async def call_llm_structured(self, model: str, messages: List[dict], response_format: BaseModel):
        pass

    @abstractmethod
    async def call_llm_tools(self, model: str, messages: List[dict], tools: List[dict], tool_choice: Union[Literal['auto', 'none'], dict] = 'auto'):
        pass

class LLMService(BaseLLMService):
    def __init__(self, name: str):
        self.name = name
        api_key, base_url = PROVIDER_INFORMATION[name]["API"]
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=2
        )

    def _get_model_id(self, model: str):
        return model, PROVIDER_INFORMATION[self.name]["MODEL_ID"][model]

    async def call_llm(self, model: str, messages: List[dict]):
        """Call the LLM with the given model and messages."""
        generic_model_name, model = self._get_model_id(model)
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content if response.choices else None
        except Exception as e:
            logger.error(f"{self.name} LLM service failed to call model {generic_model_name}: {e}")
            return None

    async def call_llm_structured(self, model: str, messages: List[dict], response_format: BaseModel):
        """Call the LLM with the given model and messages."""
        generic_model_name, model = self._get_model_id(model)
        try:
            structured_client = instructor.from_openai(self.client, mode=instructor.Mode.JSON)
            response = await structured_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_format,
            )
            return response
        except Exception as e:
            logger.error(f"{self.name} LLM service failed to call model {generic_model_name}: {e}")
            return None
    
    async def call_llm_tools(self, model: str, messages: List[dict], tools: dict[str, BaseTool], tool_choice: Union[Literal['auto', 'none'], dict] = 'auto'):
        generic_model_name, model = self._get_model_id(model)
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[tool().openai_dict for tool in tools.values()],
                tool_choice=tool_choice
            )

            if response.choices and len(response.choices) > 0 and hasattr(response.choices[0].message, 'tool_calls'):
                tool_response = response.choices[0].message.tool_calls
                chosen_tool = tools.get(tool_response[0].function.name)()
                context = await chosen_tool.run(**tool_response[0].function.arguments)
                return context
            else:
                return None
        except Exception as e:
            logger.error(f"{self.name} LLM service failed to call model {generic_model_name}: {e}")
            return None


class LocalLLMService(BaseLLMService):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def _ollama_chat(self, model: str, messages: List[dict]):
        """Low-level async wrapper for Ollama's /api/chat endpoint."""
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"Ollama returned {response.status}: {text}")
                return await response.json()

    async def call_llm(self, model: str, messages: List[dict]):
        """Call local Ollama model."""
        try:
            response = await self._ollama_chat(model, messages)
            content = response.get("message", {}).get("content")
            return content
        except Exception as e:
            logger.error(f"Local LLM service failed to call model {model}: {e}")
            return None

    async def call_llm_structured(self, model: str, messages: List[dict], response_format: BaseModel):
        """Simulate structured output by parsing into pydantic model."""
        try:
            system_instruction = {
                "role": "system",
                "content": (
                    "Respond strictly in JSON format matching this schema:\n"
                    f"{response_format.model_json_schema()}"
                )
            }
            structured_messages = [system_instruction] + messages

            response = await self._ollama_chat(model, structured_messages)
            content = response.get("message", {}).get("content")

            # Parse the response content into the provided Pydantic model
            json_start = content.find('{')
            json_str = content[json_start:]
            parsed = json.loads(json_str)
            return response_format.parse_obj(parsed)
        except Exception as e:
            logger.error(f"Local structured call failed for model {model}: {e}")
            return None

    async def call_llm_tools(
        self,
        model: str,
        messages: List[dict],
        tools: dict[str, "BaseTool"],
        tool_choice: Union[Literal['auto', 'none'], dict] = 'auto'
    ):
        """
        Very basic tool-call simulation for local models.
        Since Ollama doesn’t natively support tool-calling,
        we emulate it via JSON-instruction prompting.
        """
        try:
            tool_descriptions = {
                name: tool().description for name, tool in tools.items()
            }

            tool_prompt = (
                "You have access to the following tools. "
                "Decide which to call and provide a JSON response like "
                '{"tool": "name", "arguments": { ... }}.\n\n'
                f"{json.dumps(tool_descriptions, indent=2)}"
            )
            messages = [{"role": "system", "content": tool_prompt}] + messages

            response = await self._ollama_chat(model, messages)
            content = response.get("message", {}).get("content", "")

            json_start = content.find('{')
            parsed = json.loads(content[json_start:])

            tool_name = parsed["tool"]
            arguments = parsed.get("arguments", {})
            if tool_name in tools:
                chosen_tool = tools[tool_name]()
                result = await chosen_tool.run(**arguments)
                return result
            return None
        except Exception as e:
            logger.error(f"Local tool call failed for model {model}: {e}")
            return None