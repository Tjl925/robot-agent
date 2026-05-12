from __future__ import annotations

"""Unified LLM client wrapper for Taili agents.

This module keeps the four judgment agents stateless and model-agnostic.
Currently it targets DeepSeek-compatible OpenAI-style APIs and only needs
an API key from the environment.
"""

from dataclasses import dataclass
import json
import os
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMCallConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url_env: str = "DEEPSEEK_BASE_URL"
    model: str = "deepseek-v4-pro"
    temperature: float = 0.2
    max_retries: int = 2


class UnifiedLLMClient:
    def __init__(self, config: LLMCallConfig | None = None):
        self.config = config or LLMCallConfig()

    def _client(self) -> OpenAI:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {self.config.api_key_env}")
        base_url = os.getenv(self.config.base_url_env, "https://api.deepseek.com")
        return OpenAI(api_key=api_key, base_url=base_url)

    def generate_json(self, *, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        last_error: Exception | None = None
        client = self._client()
        schema_name = schema.__name__
        for _ in range(max(1, self.config.max_retries + 1)):
            try:
                response = client.responses.create(
                    model=self.config.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    text={"format": {"type": "json_object"}},
                )
                text = getattr(response, "output_text", None)
                if not text:
                    output = getattr(response, "output", None)
                    text = ""
                    if output:
                        try:
                            text = json.dumps(output, ensure_ascii=False)
                        except Exception:
                            text = str(output)
                return schema.model_validate_json(text)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to call DeepSeek model for {schema_name}: {last_error}")
