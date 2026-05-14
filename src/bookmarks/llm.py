from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional


class LLMClient:
    def __init__(
        self,
        model: str,
        eval_model: str,
        disc_model: str,
        cache_file: str,
        temperature: float = 1e-8,
        max_retries: int = 3,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.eval_model = eval_model
        self.disc_model = disc_model
        self.cache_file = Path(cache_file)
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache: dict[str, str] = {}
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    def _cache_key(self, prompt: str, model: str) -> str:
        return f"prompt: {prompt} model: {model}"

    def _load_cache(self) -> None:
        if not self.cache_file.exists():
            return
        with self.cache_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = row.get("prompt", "")
                model = row.get("model") or row.get("engine", "")
                response = row.get("response", "")
                if prompt and model and response:
                    self.cache[self._cache_key(prompt, model)] = response

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or self.model
        key = self._cache_key(prompt, model)
        if key in self.cache:
            return self.cache[key]

        last_error: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=model,
                    temperature=self.temperature,
                    input=[{"role": "user", "content": prompt}],
                ).output_text
                self.cache[key] = response
                with self.cache_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"prompt": prompt, "response": response, "model": model}, ensure_ascii=False) + "\n")
                return response
            except Exception as exc:  # keep retry behavior close to the notebook without hanging forever
                last_error = exc
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_error
