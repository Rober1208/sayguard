"""Client for an Ollama-compatible chat API.

The same calls work against Ollama on a PC and against hailo-ollama serving
Qwen2.5-1.5B-Instruct on the UGen300 (Hailo GenAI Model Zoo), so moving the
LLM onto the accelerator is a base_url change.
"""
from __future__ import annotations

import httpx


class LLMClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_s

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def chat(self, messages: list[dict], fmt: dict | str | None = None,
             temperature: float = 0.2, max_tokens: int = 512) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if fmt is not None:
            body["format"] = fmt
        r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]
