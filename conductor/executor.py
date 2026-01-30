"""
Executor for Conductor.

Dispatches requests to LLM providers and handles responses.
This module is designed to be pluggable - you can use real providers
or mock providers for testing.
"""

import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Optional, Any
import uuid

from conductor.schemas import (
    Request,
    Decision,
    Outcome,
    FinishReason,
    Provider,
)


@dataclass
class ExecutionResult:
    """Raw result from LLM execution."""
    response_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    finish_reason: FinishReason
    http_status: int
    raw_response: Optional[dict] = None
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def execute(
        self,
        request: Request,
        decision: Decision,
    ) -> ExecutionResult:
        """Execute a request with the given decision parameters."""
        pass


class MockProvider(LLMProvider):
    """
    Mock provider for testing.
    
    Simulates LLM responses with configurable behavior.
    """
    
    def __init__(
        self,
        latency_ms: int = 500,
        failure_rate: float = 0.0,
        output_tokens_multiplier: float = 1.0,
    ):
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.output_tokens_multiplier = output_tokens_multiplier
    
    def execute(
        self,
        request: Request,
        decision: Decision,
    ) -> ExecutionResult:
        """Execute with simulated latency and response."""
        import random
        
        # Simulate latency
        actual_latency = int(self.latency_ms * (0.8 + random.random() * 0.4))
        time.sleep(actual_latency / 1000)
        
        # Simulate failure
        if random.random() < self.failure_rate:
            return ExecutionResult(
                response_text="",
                input_tokens=len(request.prompt) // 4,
                output_tokens=0,
                latency_ms=actual_latency,
                finish_reason=FinishReason.ERROR,
                http_status=500,
                error="Simulated failure",
            )
        
        # Generate mock response
        input_tokens = len(request.prompt) // 4
        output_tokens = int(100 * self.output_tokens_multiplier)
        
        return ExecutionResult(
            response_text=f"[Mock response for {decision.strategy_id}]",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=actual_latency,
            finish_reason=FinishReason.STOP,
            http_status=200,
        )


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.
    
    Requires OPENAI_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
    
    def execute(
        self,
        request: Request,
        decision: Decision,
    ) -> ExecutionResult:
        """Execute request via OpenAI API."""
        start_time = time.time()
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=decision.model_id,
                messages=messages,
                temperature=decision.temperature,
                max_tokens=decision.max_tokens,
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Map finish reason
            finish_reason_map = {
                "stop": FinishReason.STOP,
                "length": FinishReason.LENGTH,
                "tool_calls": FinishReason.TOOL_CALL,
            }
            finish_reason = finish_reason_map.get(
                response.choices[0].finish_reason,
                FinishReason.STOP
            )
            
            return ExecutionResult(
                response_text=response.choices[0].message.content or "",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                http_status=200,
                raw_response=response.model_dump(),
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                response_text="",
                input_tokens=len(request.prompt) // 4,
                output_tokens=0,
                latency_ms=latency_ms,
                finish_reason=FinishReason.ERROR,
                http_status=500,
                error=str(e),
            )


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider.
    
    Requires ANTHROPIC_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client
    
    def execute(
        self,
        request: Request,
        decision: Decision,
    ) -> ExecutionResult:
        """Execute request via Anthropic API."""
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=decision.model_id,
                max_tokens=decision.max_tokens,
                system=request.system_prompt or "",
                messages=[{"role": "user", "content": request.prompt}],
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Map stop reason
            finish_reason = FinishReason.STOP
            if response.stop_reason == "max_tokens":
                finish_reason = FinishReason.LENGTH
            
            return ExecutionResult(
                response_text=response.content[0].text if response.content else "",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                http_status=200,
                raw_response=response.model_dump(),
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                response_text="",
                input_tokens=len(request.prompt) // 4,
                output_tokens=0,
                latency_ms=latency_ms,
                finish_reason=FinishReason.ERROR,
                http_status=500,
                error=str(e),
            )


class Executor:
    """
    Main executor that dispatches to the appropriate provider.
    """

    def __init__(
        self,
        providers: Optional[dict[Provider, LLMProvider]] = None,
        default_provider: Optional[LLMProvider] = None,
        max_retries: int = 3,
        retry_base_delay_ms: int = 100,
        retry_max_delay_ms: int = 10000,
    ):
        """
        Initialize executor.

        Args:
            providers: LLM providers by type.
            default_provider: Fallback provider (defaults to MockProvider).
            max_retries: Maximum retry attempts for failed requests.
            retry_base_delay_ms: Base delay for exponential backoff (milliseconds).
            retry_max_delay_ms: Maximum delay between retries (milliseconds).
        """
        self.providers = providers or {}
        self.default_provider = default_provider or MockProvider()
        self.max_retries = max_retries
        self.retry_base_delay_ms = retry_base_delay_ms
        self.retry_max_delay_ms = retry_max_delay_ms
    
    def execute(
        self,
        request: Request,
        decision: Decision,
    ) -> Outcome:
        """
        Execute a request and return the outcome.
        
        Args:
            request: Original request.
            decision: Routing decision.
        
        Returns:
            Outcome with actual measurements.
        """
        # Determine provider from strategy
        provider_type = self._get_provider_type(decision.model_id)
        provider = self.providers.get(provider_type, self.default_provider)

        # Execute with exponential backoff retries
        retry_count = 0
        result = None
        last_error = None

        while retry_count <= self.max_retries:
            result = provider.execute(request, decision)

            # Success - return immediately
            if result.finish_reason != FinishReason.ERROR:
                break

            last_error = result.error
            retry_count += 1

            # If we're going to retry, wait with exponential backoff
            if retry_count <= self.max_retries:
                delay_ms = min(
                    self.retry_base_delay_ms * (2 ** (retry_count - 1)),
                    self.retry_max_delay_ms
                )
                time.sleep(delay_ms / 1000)
        
        # Compute actual cost
        actual_cost = self._compute_actual_cost(
            decision.model_id,
            result.input_tokens,
            result.output_tokens,
            decision.verifier_enabled,
            decision.tools_enabled,
        )
        
        # Compute quality proxy score
        quality_proxy = self._compute_quality_proxy(
            result,
            request,
        )
        
        # Check if SLA was met
        # (This is a simplification - in reality you'd check against the prediction)
        sla_met = result.finish_reason == FinishReason.STOP
        
        # Compute prediction errors
        cost_error_pct = (
            (actual_cost - decision.predicted_cost_usd) / 
            decision.predicted_cost_usd * 100
            if decision.predicted_cost_usd > 0 else 0
        )
        latency_error_pct = (
            (result.latency_ms - decision.predicted_latency_p50_ms) /
            decision.predicted_latency_p50_ms * 100
            if decision.predicted_latency_p50_ms > 0 else 0
        )
        
        return Outcome(
            outcome_id=str(uuid.uuid4()),
            decision_id=decision.decision_id,
            request_id=decision.request_id,
            timestamp_utc=datetime.now(UTC),
            
            actual_cost_usd=actual_cost,
            actual_latency_ms=result.latency_ms,
            actual_input_tokens=result.input_tokens,
            actual_output_tokens=result.output_tokens,
            
            finish_reason=result.finish_reason,
            http_status=result.http_status,
            retry_count=retry_count,
            verifier_passed=None,  # Would be set by verifier
            format_valid=self._check_format_valid(result, request),
            
            quality_proxy_score=quality_proxy,
            sla_met=sla_met,
            cost_error_pct=cost_error_pct,
            latency_error_pct=latency_error_pct,
            
            response_text=result.response_text,
        )
    
    def _get_provider_type(self, model_id: str) -> Provider:
        """Determine provider type from model ID."""
        if model_id.startswith("gpt-") or model_id.startswith("o1"):
            return Provider.OPENAI
        elif model_id.startswith("claude-"):
            return Provider.ANTHROPIC
        else:
            return Provider.SELF_HOSTED
    
    def _compute_actual_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        verifier_enabled: bool,
        tools_enabled: bool,
    ) -> float:
        """Compute actual cost from token usage."""
        # Pricing (should match coefficients)
        pricing = {
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-4o": (0.0025, 0.01),
            "claude-3-haiku-20240307": (0.00025, 0.00125),
            "claude-3-5-sonnet-20241022": (0.003, 0.015),
        }
        
        input_price, output_price = pricing.get(model_id, (0.001, 0.002))
        
        cost = (input_tokens / 1000) * input_price
        cost += (output_tokens / 1000) * output_price
        
        if verifier_enabled:
            cost += 0.002
        if tools_enabled:
            cost += 0.0005
        
        return round(cost, 6)
    
    def _compute_quality_proxy(
        self,
        result: ExecutionResult,
        request: Request,
    ) -> float:
        """
        Compute quality proxy score (0.0 to 1.0).
        
        Based on observable signals:
        - Finish reason
        - Response length
        - Format validity (if applicable)
        """
        score = 1.0
        
        # Penalty for non-stop finish
        if result.finish_reason == FinishReason.LENGTH:
            score -= 0.2  # Truncated
        elif result.finish_reason == FinishReason.ERROR:
            score -= 0.5
        elif result.finish_reason == FinishReason.TIMEOUT:
            score -= 0.4
        
        # Penalty for very short response (might be incomplete)
        if result.output_tokens < 10:
            score -= 0.1
        
        # Penalty for retry
        # (This would need retry_count passed in)
        
        return max(0.0, min(1.0, score))
    
    def _check_format_valid(
        self,
        result: ExecutionResult,
        request: Request,
    ) -> bool:
        """Check if response format is valid (for JSON tasks)."""
        from conductor.schemas import TaskType
        
        if request.task_type != TaskType.EXTRACT_JSON:
            return True
        
        try:
            json.loads(result.response_text)
            return True
        except (json.JSONDecodeError, TypeError):
            # Try to extract JSON from response
            text = result.response_text
            if "{" in text and "}" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                try:
                    json.loads(text[start:end])
                    return True
                except:
                    pass
            return False
