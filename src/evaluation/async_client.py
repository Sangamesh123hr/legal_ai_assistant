"""
Async DeepSeek API Client

High-performance async client for DeepSeek API with streaming support.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass

import httpx

from .config import EvalConfig, ModelResponse, DEEPSEEK_PRICING

logger = logging.getLogger(__name__)


class DeepSeekAsyncClient:
    """
    Async client for DeepSeek API.

    Supports:
    - deepseek-chat (V3)
    - deepseek-reasoner (R1) with reasoning extraction
    - Streaming responses
    - Automatic cost tracking
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.timeout = config.timeout

        # Create async HTTP client
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def chat(
        self,
        model: str,
        messages: list,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> ModelResponse:
        """
        Send a chat completion request.

        Args:
            model: Model ID (deepseek-chat or deepseek-reasoner)
            messages: List of message dicts
            system_prompt: Optional system message
            stream: Whether to stream the response

        Returns:
            ModelResponse with content, tokens, latency, and cost
        """
        start_time = time.time()

        # Build messages with system prompt
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        payload = {
            "model": model,
            "messages": all_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }

        # Add reasoning parameters for R1
        if model == "deepseek-reasoner":
            payload["stream"] = False  # R1 doesn't support streaming

        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            # Extract response
            choice = data["choices"][0]

            # Check for reasoning content (R1 model)
            reasoning = None
            if model == "deepseek-reasoner" and "reasoning_content" in choice.get(
                "message", {}
            ):
                reasoning = choice["message"]["reasoning_content"]

            content = choice["message"]["content"]

            # Extract token usage
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Calculate cost
            pricing = DEEPSEEK_PRICING.get(model)
            if pricing:
                cost = pricing.calculate_cost(input_tokens, output_tokens)
            else:
                cost = 0.0

            return ModelResponse(
                content=content,
                reasoning=reasoning,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return ModelResponse(
                content="",
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Request error: {e}")
            return ModelResponse(
                content="",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def batch_chat(
        self,
        model: str,
        requests: list,
        max_concurrent: int = 10,
    ) -> list:
        """
        Process multiple chat requests concurrently.

        Args:
            model: Model ID
            requests: List of (messages, system_prompt) tuples
            max_concurrent: Maximum concurrent requests

        Returns:
            List of ModelResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_chat(messages, system_prompt):
            async with semaphore:
                return await self.chat(model, messages, system_prompt)

        tasks = [
            bounded_chat(req[0], req[1] if len(req) > 1 else None) for req in requests
        ]

        return await asyncio.gather(*tasks)


async def create_client(config: EvalConfig) -> DeepSeekAsyncClient:
    """Create and return an async DeepSeek client."""
    client = DeepSeekAsyncClient(config)
    await client.__aenter__()
    return client
