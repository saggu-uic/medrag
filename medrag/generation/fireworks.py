from __future__ import annotations

import os
import time
import requests


class FireworksGenerator:
    """
    Generates answers using Fireworks AI API (Llama 3.1 8B Instruct).
    Reads FIREWORKS_API_KEY from environment or .env file.
    Includes retry logic for rate limiting (429 errors).
    """

    def __init__(
        self,
        model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
        max_tokens: int = 256,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 10.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"

        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("FIREWORKS_API_KEY")
            except ImportError:
                pass
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY not set. Add it to .env or environment.")
        self._api_key = api_key

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        context = "\n\n".join(f"[{i+1}] {c['text'][:400]}" for i, c in enumerate(context_chunks[:5]))
        prompt = (
            "You are a medical information assistant. "
            "Answer the question based on the context provided below. "
            "Be concise and factual. "
            "If the context is insufficient, use what is available and indicate uncertainty.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.url, json=payload, headers=headers, timeout=30)
                if response.status_code == 429:
                    wait = self.retry_delay * (attempt + 1)
                    print(f"  Rate limited. Waiting {wait}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"  HTTP error: {e}. Retrying...")
                time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed after {self.max_retries} retries")
