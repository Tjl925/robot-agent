from __future__ import annotations

"""Unified LLM client wrapper for Taili agents.

This module keeps the four judgment agents stateless and model-agnostic.
Currently it targets DeepSeek-compatible OpenAI-style APIs and only needs
an API key from the environment.
"""

from dataclasses import dataclass
import json
import os
import sys
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMCallConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url_env: str = "DEEPSEEK_BASE_URL"
    default_base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    max_retries: int = 2


class UnifiedLLMClient:
    def __init__(self, config: LLMCallConfig | None = None):
        self.config = config or LLMCallConfig()

    def _client(self) -> OpenAI:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {self.config.api_key_env}")
        base_url = os.getenv(self.config.base_url_env, self.config.default_base_url)
        return OpenAI(api_key=api_key, base_url=base_url)

    def generate_json(self, *, system_prompt: str, user_prompt: str | list, schema: type[T]) -> T:
        last_error: Exception | None = None
        client = self._client()
        schema_name = schema.__name__
        for _ in range(max(1, self.config.max_retries + 1)):
            try:
                kwargs = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "stream": True
                }
                
                # 根据不同模型附加对应的高级推理参数
                if "deepseek" in self.config.model.lower():
                    kwargs["reasoning_effort"] = "high"
                    kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                elif "qwen" in self.config.model.lower():
                    kwargs["extra_body"] = {"enable_thinking": True}
                    
                response = client.chat.completions.create(**kwargs)
                
                print(f"\n\033[96m[{schema_name} LLM Response Stream]\033[0m")
                full_text = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    # For deepseek-reasoner reasoning chunks
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        print(f"\033[90m{delta.reasoning_content}\033[0m", end="", flush=True)
                    # For normal content chunks
                    if hasattr(delta, "content") and delta.content:
                        full_text += delta.content
                print("\n")
                return schema.model_validate_json(full_text)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to call DeepSeek model for {schema_name}: {last_error}")
