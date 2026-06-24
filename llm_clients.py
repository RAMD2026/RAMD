# llm_clients.py
import os
import time
from typing import Dict, Any
import math
import json


class BaseLLMClient:
    def __init__(self, model_name: str, temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def __init__(self, model_name: str, temperature: float, env_var: str,
                 max_retries: int = 6, base_delay: float = 2.0):
        super().__init__(model_name, temperature)
        from openai import OpenAI
        api_key = os.getenv(env_var)
        if not api_key:
            raise RuntimeError(f"Missing OpenAI API key in env var {env_var}")
        self.client = OpenAI(api_key=api_key)
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    def _has_surrogate(self, text: str) -> bool:
        return any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from openai import RateLimitError

        system_prompt = self._clean_text(system_prompt)
        user_prompt = self._clean_text(user_prompt)

        if self._has_surrogate(system_prompt):
            print("WARNING: system_prompt still has surrogate chars")
        if self._has_surrogate(user_prompt):
            print("WARNING: user_prompt still has surrogate chars")

        payload = {
            "model": str(self.model_name),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # local JSON check
        json.dumps(payload, allow_nan=False)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**payload)
                return response.choices[0].message.content or ""
            except RateLimitError as e:
                last_err = e
                retry_after = None
                try:
                    retry_after = float(e.response.headers.get("retry-after", 0))
                except Exception:
                    pass
                delay = retry_after if retry_after else self.base_delay * (2 ** attempt)
                print(f"[RateLimit] attempt {attempt + 1}/{self.max_retries}, "
                      f"sleeping {delay:.1f}s before retry...")
                time.sleep(delay)
            except Exception as e:
                print("===== OPENAI REQUEST FAILED =====")
                print("model:", repr(payload["model"]))
                print("system len:", len(system_prompt))
                print("user len:", len(user_prompt))
                print("system preview:", repr(system_prompt[:500]))
                print("user preview:", repr(user_prompt[:500]))
                raise

        print(f"===== OPENAI RATE LIMIT: exhausted {self.max_retries} retries =====")
        raise last_err


class AnthropicClient(BaseLLMClient):
    def __init__(self, model_name: str, temperature: float, env_var: str):
        super().__init__(model_name, temperature)
        import anthropic
        api_key = os.getenv(env_var)
        if not api_key:
            raise RuntimeError(f"Missing Anthropic API key in env var {env_var}")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content)


class MistralClient(BaseLLMClient):
    def __init__(self, model_name: str, temperature: float, env_var: str):
        super().__init__(model_name, temperature)
        from mistralai import Mistral
        api_key = os.getenv(env_var)
        if not api_key:
            raise RuntimeError(f"Missing Mistral API key in env var {env_var}")
        self.client = Mistral(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.complete(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


def build_llm_client(model_cfg: Dict[str, Any], api_keys_cfg: Dict[str, Any]) -> BaseLLMClient:
    provider = model_cfg["provider"]
    model_name = model_cfg["model"]
    temperature = float(model_cfg.get("temperature", 0.2))

    if provider == "openai":
        return OpenAIClient(model_name, temperature, api_keys_cfg["openai_env"])
    elif provider == "anthropic":
        return AnthropicClient(model_name, temperature, api_keys_cfg["anthropic_env"])
    elif provider == "mistral":
        return MistralClient(model_name, temperature, api_keys_cfg["mistral_env"])
    else:
        raise ValueError(f"Unknown provider: {provider}")