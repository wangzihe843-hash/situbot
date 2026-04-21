#!/usr/bin/env python3
"""DashScope/OpenAI-compatible LLM client for Qwen models."""

import json
import time
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class DashScopeClient:
    """Client for DashScope API (OpenAI-compatible endpoint).

    Supports Qwen-Plus, Qwen-Max, and other models available via DashScope.
    """

    def __init__(self, endpoint: str, api_key: str, model: str = "qwen-plus",
                 temperature: float = 0.3, max_tokens: int = 2048,
                 timeout: int = 30, max_retries: int = 3):
        """
        Args:
            endpoint: API base URL (e.g., https://dashscope.aliyuncs.com/compatible-mode/v1).
            api_key: DashScope API key.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on transient failures.
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None

    def _get_session(self):
        """Lazy-init requests session."""
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
        """Send a chat completion request.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            response_format: Optional format spec (e.g., {"type": "json_object"}).

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If all retries fail.
        """
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
                    # Rate limited — back off
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
        """Chat and parse response as JSON.

        Automatically adds response_format for JSON mode and retries
        on parse failures.
        """
        kwargs.setdefault("response_format", {"type": "json_object"})

        last_error = None
        for attempt in range(_json_retries):
            response = self.chat(messages, **kwargs)

            # Extract JSON from response, handling markdown fences and surrounding text
            text = response.strip()

            # Try extracting from ```json ... ``` or ``` ... ``` fences first
            import re
            fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
            if fence_match:
                text = fence_match.group(1).strip()
            else:
                # Try to find the outermost JSON object or array
                for i, ch in enumerate(text):
                    if ch in ('{', '['):
                        text = text[i:]
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
