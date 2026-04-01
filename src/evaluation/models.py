"""
Model Wrappers for LLM Evaluation
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

from .config import ModelConfig, ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Standardized response from any LLM."""

    content: str
    model: str
    provider: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    raw_response: Optional[Any] = None

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "latency_seconds": self.latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None

    @abstractmethod
    def _call_api(self, prompt: str, **kwargs) -> ModelResponse:
        """Make the actual API call. Override in subclasses."""
        pass

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> ModelResponse:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            **kwargs: Additional model-specific parameters

        Returns:
            ModelResponse with content and metadata
        """
        start_time = time.time()

        try:
            response = self._call_api(prompt, system_prompt, **kwargs)
            response.latency_seconds = time.time() - start_time

            # Calculate cost
            response.cost_usd = self.config.estimate_cost(
                response.input_tokens, response.output_tokens
            )

            return response

        except Exception as e:
            logger.error(f"Error calling {self.config.name}: {e}")
            return ModelResponse(
                content=f"ERROR: {str(e)}",
                model=self.config.model_id,
                provider=self.config.provider.value,
                latency_seconds=time.time() - start_time,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            )

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[ModelResponse]:
        """Generate responses for multiple prompts."""
        responses = []
        iterator = (
            tqdm(prompts, desc=f"Evaluating {self.config.name}")
            if show_progress
            else prompts
        )

        for prompt in iterator:
            response = self.generate(prompt, system_prompt, **kwargs)
            responses.append(response)

            if show_progress and isinstance(iterator, tqdm):
                iterator.set_postfix({"tokens": response.output_tokens})

        return responses


class ClaudeWrapper(ModelWrapper):
    """Wrapper for Anthropic Claude models."""

    def _call_api(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> ModelResponse:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        if self._client is None:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise ValueError(f"Missing API key: {self.config.api_key_env}")
            self._client = Anthropic(api_key=api_key)

        messages = [{"role": "user", "content": prompt}]

        response = self._client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=messages,
            **kwargs,
        )

        # Estimate tokens (Claude doesn't return exact counts in older APIs)
        input_tokens = self._estimate_tokens(prompt + (system_prompt or ""))
        output_tokens = self._estimate_tokens(response.content[0].text)

        return ModelResponse(
            content=response.content[0].text,
            model=self.config.model_id,
            provider=ModelProvider.ANTHROPIC.value,
            latency_seconds=0,  # Set by generate()
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0,  # Set by generate()
            raw_response=response,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 chars per token for English."""
        return max(1, len(text) // 4)


class DeepSeekWrapper(ModelWrapper):
    """Wrapper for DeepSeek models."""

    def _call_api(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> ModelResponse:
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        if self._client is None:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise ValueError(f"Missing API key: {self.config.api_key_env}")
            self._client = openai.OpenAI(
                api_key=api_key, base_url="https://api.deepseek.com"
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completion.create(
            model=self.config.model_id,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            **kwargs,
        )

        usage = response.usage
        return ModelResponse(
            content=response.choices[0].message.content,
            model=self.config.model_id,
            provider=ModelProvider.DEEPSEEK.value,
            latency_seconds=0,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=0,
            raw_response=response,
        )


class GPTWrapper(ModelWrapper):
    """Wrapper for OpenAI GPT models."""

    def _call_api(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> ModelResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        if self._client is None:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise ValueError(f"Missing API key: {self.config.api_key_env}")
            self._client = OpenAI(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completion.create(
            model=self.config.model_id,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            **kwargs,
        )

        usage = response.usage
        return ModelResponse(
            content=response.choices[0].message.content,
            model=self.config.model_id,
            provider=ModelProvider.OPENAI.value,
            latency_seconds=0,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=0,
            raw_response=response,
        )


def create_model_wrapper(config: ModelConfig) -> ModelWrapper:
    """Factory function to create model wrapper based on provider."""
    wrappers = {
        ModelProvider.ANTHROPIC: ClaudeWrapper,
        ModelProvider.DEEPSEEK: DeepSeekWrapper,
        ModelProvider.OPENAI: GPTWrapper,
        ModelProvider.GOOGLE: GPTWrapper,  # Use same interface
    }

    wrapper_class = wrappers.get(config.provider)
    if not wrapper_class:
        raise ValueError(f"Unsupported provider: {config.provider}")

    return wrapper_class(config)
