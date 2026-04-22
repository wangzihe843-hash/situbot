#!/usr/bin/env python3
"""DashScope/OpenAI-compatible LLM client for Qwen models."""

import json
import time
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class DashScopeClient:
    """Client for DashScope API (OpenAI-compatible endpoint)."""

    def __init__(self, endpoint: str, api_key: str, model: str = "qwen-plus",
                 temperature: float = 0.3, max_tokens: int = 2048,
                 timeout: int = 30, max_retries: int = 3):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
        return self._session

    def chat(self, messages: List[Dict[str, str]],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             response_format: Optional[Dict] = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        url = f"{self.endpoint}/chat/completions"
        session = self._get_session()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 2, 30)
                    logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Request failed: {e}, retrying in {wait}s")
                    time.sleep(wait)

        raise RuntimeError(f"All {self.max_retries} attempts failed. Last error: {last_error}")

    def chat_json(self, messages: List[Dict[str, str]],
                   _json_retries: int = 2, **kwargs) -> Dict[str, Any]:
        kwargs.setdefault("response_format", {"type": "json_object"})

        last_error = None
        for attempt in range(_json_retries):
            response = self.chat(messages, **kwargs)

            text = response.strip()

            import re
            fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
            if fence_match:
                text = fence_match.group(1).strip()
            else:
                start_ch = None
                for i, ch in enumerate(text):
                    if ch in ('{', '['):
                        start_ch = ch
                        text = text[i:]
                        break
                if start_ch:
                    close_ch = '}' if start_ch == '{' else ']'
                    depth = 0
                    in_string = False
                    escape = False
                    for j, c in enumerate(text):
                        if escape:
                            escape = False
                            continue
                        if c == '\\' and in_string:
                            escape = True
                            continue
                        if c == '"' and not escape:
                            in_string = not in_string
                            continue
                        if in_string:
                            continue
                        if c == start_ch:
                            depth += 1
                        elif c == close_ch:
                            depth -= 1
                            if depth == 0:
                                text = text[:j + 1]
                                break

            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"JSON parse attempt {attempt+1}/{_json_retries} failed: {e}")
                if attempt < _json_retries - 1:
                    logger.info("Retrying LLM call for valid JSON...")

        logger.error(f"Failed to parse JSON after {_json_retries} attempts. Raw: {response[:500]}")
        raise last_error
