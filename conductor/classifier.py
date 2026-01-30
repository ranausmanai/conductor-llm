"""
Request Classifier for Conductor.

Analyzes incoming requests to extract features used for routing decisions.
"""

import re
from conductor.schemas import (
    Request,
    RequestFeatures,
    TaskType,
    UrgencyTier,
)
from conductor.coefficients import Coefficients, DEFAULT_COEFFICIENTS


class Classifier:
    """
    Extracts features from raw requests for routing decisions.
    
    Features include:
    - Token counts (input and estimated output)
    - Complexity score
    - Urgency tier
    - Structured output detection
    """
    
    def __init__(self, coefficients: Coefficients = None):
        self.coefficients = coefficients or DEFAULT_COEFFICIENTS
        
        # Patterns for complexity detection
        self._question_pattern = re.compile(r'\?')
        self._negation_pattern = re.compile(
            r'\b(not|no|never|don\'t|doesn\'t|won\'t|can\'t|shouldn\'t|'
            r'wouldn\'t|couldn\'t|mustn\'t|without|except|unless)\b',
            re.IGNORECASE
        )
        self._reasoning_pattern = re.compile(
            r'\b(explain|why|how|step.by.step|reason|because|therefore|'
            r'analyze|compare|contrast|evaluate|assess)\b',
            re.IGNORECASE
        )
        self._code_pattern = re.compile(r'[{}\[\]<>/\\|&;]')
    
    def classify(self, request: Request) -> RequestFeatures:
        """
        Extract features from a request.
        
        Args:
            request: The incoming request.
        
        Returns:
            RequestFeatures with extracted metrics.
        """
        # Count input tokens (rough estimate: ~4 chars per token)
        text = request.prompt + (request.system_prompt or "")
        input_token_count = self._estimate_tokens(text)
        
        # Estimate output tokens
        estimated_output_tokens = self._estimate_output_tokens(
            request.task_type,
            input_token_count,
            request.expected_output_tokens,
        )
        
        # Compute complexity score
        complexity_score = self._compute_complexity(request.prompt, input_token_count)
        
        # Determine urgency tier from latency constraint
        urgency_tier = self._determine_urgency(request.max_latency_ms)
        
        # Detect structured output
        has_structured_output = self._detect_structured_output(request)
        
        return RequestFeatures(
            request_id=request.request_id,
            task_type=request.task_type,
            input_token_count=input_token_count,
            estimated_output_tokens=estimated_output_tokens,
            complexity_score=complexity_score,
            urgency_tier=urgency_tier,
            has_structured_output=has_structured_output,
            max_latency_ms=request.max_latency_ms,
            latency_percentile=request.latency_percentile,
            min_quality_score=request.min_quality_score,
            max_cost_usd=request.max_cost_usd,
            allow_batching=request.allow_batching,
            require_tools=request.require_tools,
            require_verifier=request.require_verifier,
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        
        Uses a simple heuristic of ~4 characters per token.
        For production, use tiktoken or the model's tokenizer.
        """
        # Rough estimate: ~4 chars per token for English
        # This is a reasonable approximation for GPT models
        return max(1, len(text) // 4)
    
    def _estimate_output_tokens(
        self,
        task_type: TaskType,
        input_token_count: int,
        client_estimate: int | None,
    ) -> int:
        """
        Estimate output tokens based on task type.
        
        Priority:
        1. Client-provided estimate (if given)
        2. Task-specific formula
        """
        if client_estimate is not None:
            return client_estimate
        
        task_coeffs = self.coefficients.get_task_coeffs(task_type.value)
        
        if task_type == TaskType.CLASSIFY:
            return task_coeffs.base_output_tokens  # ~10 tokens
        
        elif task_type == TaskType.EXTRACT_JSON:
            return task_coeffs.base_output_tokens  # ~150 tokens
        
        elif task_type == TaskType.SUMMARIZE:
            # Summarization: typically 20-30% of input length
            return min(max(50, int(input_token_count * 0.25)), 1000)
        
        elif task_type == TaskType.REWRITE:
            # Rewrite: similar length to input
            return int(input_token_count * 1.1)
        
        elif task_type == TaskType.GENERATE_LONG:
            # Long-form: use a reasonable default
            return 1500
        
        elif task_type == TaskType.CHAT:
            return task_coeffs.base_output_tokens  # ~200 tokens
        
        return 200  # Default fallback
    
    def _compute_complexity(self, prompt: str, input_token_count: int) -> float:
        """
        Compute complexity score (0.0 to 1.0).
        
        Higher score = more complex = higher risk of quality issues.
        
        Factors:
        - Length
        - Special characters (code/JSON indicators)
        - Number of questions
        - Negation words
        - Reasoning keywords
        """
        score = 0.0
        
        # Length factor (up to 0.2)
        length_factor = min(input_token_count / 5000, 1.0) * 0.2
        score += length_factor
        
        # Special characters factor (up to 0.2)
        special_chars = len(self._code_pattern.findall(prompt))
        special_factor = min(special_chars / 50, 1.0) * 0.2
        score += special_factor
        
        # Question count factor (up to 0.15)
        question_count = len(self._question_pattern.findall(prompt))
        question_factor = min(question_count / 3, 1.0) * 0.15
        score += question_factor
        
        # Negation factor (up to 0.15)
        negation_matches = len(self._negation_pattern.findall(prompt))
        negation_factor = min(negation_matches / 5, 1.0) * 0.15
        score += negation_factor
        
        # Reasoning keywords factor (up to 0.15)
        reasoning_matches = len(self._reasoning_pattern.findall(prompt))
        reasoning_factor = min(reasoning_matches / 3, 1.0) * 0.15
        score += reasoning_factor
        
        # Multi-part instructions (up to 0.15)
        # Look for numbered lists or bullet points
        numbered = len(re.findall(r'^\s*\d+[.)]\s', prompt, re.MULTILINE))
        bulleted = len(re.findall(r'^\s*[-*•]\s', prompt, re.MULTILINE))
        list_items = numbered + bulleted
        multipart_factor = min(list_items / 5, 1.0) * 0.15
        score += multipart_factor
        
        return min(score, 1.0)
    
    def _determine_urgency(self, max_latency_ms: int) -> UrgencyTier:
        """
        Determine urgency tier from latency constraint.
        """
        if max_latency_ms < 500:
            return UrgencyTier.REALTIME
        elif max_latency_ms < 2000:
            return UrgencyTier.INTERACTIVE
        else:
            return UrgencyTier.BATCH
    
    def _detect_structured_output(self, request: Request) -> bool:
        """
        Detect if the request expects structured output (JSON, XML, etc.).
        """
        if request.task_type == TaskType.EXTRACT_JSON:
            return True
        
        prompt_lower = request.prompt.lower()
        
        # Check for JSON-related keywords
        json_indicators = [
            'json', 'xml', 'yaml', 'csv',
            'format:', 'schema:', 'structure:',
            '{', '}',
        ]
        
        return any(indicator in prompt_lower for indicator in json_indicators)
