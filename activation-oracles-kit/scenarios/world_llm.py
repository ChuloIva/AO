"""
World LLM - OpenRouter-based simulation engine for scenario environments
"""

import requests
from typing import List, Dict, Optional
import time
import re


class WorldLLMResponse:
    """
    Unified response from World LLM with type information.

    The World LLM can respond in different modes:
    - patient_dialogue: Patient speaking naturally
    - tool_result: Test results or game mechanics
    - evaluation: Diagnosis evaluation
    """

    def __init__(self, content: str, response_type: str, metadata: Optional[Dict] = None):
        """
        Initialize World LLM Response.

        Args:
            content: The actual response text
            response_type: Type of response ("patient_dialogue", "tool_result", "evaluation")
            metadata: Additional metadata (score changes, costs, etc.)
        """
        self.content = content
        self.response_type = response_type
        self.metadata = metadata or {}

    def __str__(self):
        return self.content


class WorldLLM:
    """
    OpenRouter-based World LLM for scenario simulation.
    Uses OpenRouter API to provide flexible model selection for environment simulation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    ):
        """
        Initialize World LLM.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        retry_count: int = 3,
        use_unified_response: bool = True
    ) -> 'WorldLLMResponse':
        """
        Generate response from World LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt for world simulation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            retry_count: Number of retries on failure
            use_unified_response: If True, parse and return WorldLLMResponse. If False, return raw string (backward compatibility)

        Returns:
            WorldLLMResponse object with parsed type and metadata

        Raises:
            RuntimeError: If API call fails after retries
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",  # Streamlit default
            "X-Title": "Activation Oracles Research Kit"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt}
            ] + messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()

                # Parse response
                result = response.json()
                raw_response = result["choices"][0]["message"]["content"]

                # Parse response type and metadata
                if use_unified_response:
                    response_type = self._determine_response_type(raw_response)
                    metadata = self._extract_metadata(raw_response, response_type)

                    return WorldLLMResponse(
                        content=raw_response,
                        response_type=response_type,
                        metadata=metadata
                    )
                else:
                    # Backward compatibility: return raw string wrapped in WorldLLMResponse
                    return WorldLLMResponse(
                        content=raw_response,
                        response_type="raw",
                        metadata={}
                    )

            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                time.sleep(2 ** attempt)  # Exponential backoff

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code == 429:  # Rate limit
                    last_error = "Rate limited"
                    time.sleep(5 * (attempt + 1))  # Longer backoff for rate limits
                elif status_code >= 500:  # Server error
                    last_error = f"Server error ({status_code})"
                    time.sleep(2 ** attempt)
                else:
                    # Client error - don't retry
                    raise RuntimeError(f"World LLM API error ({status_code}): {e.response.text}")

            except Exception as e:
                last_error = str(e)
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)

        # All retries failed
        raise RuntimeError(f"World LLM API call failed after {retry_count} attempts: {last_error}")

    def _determine_response_type(self, response: str) -> str:
        """
        Infer response type from content based on markers.

        Args:
            response: Raw response from LLM

        Returns:
            Response type: "tool_result", "evaluation", or "patient_dialogue"
        """
        response_upper = response.upper()

        # Check for explicit markers
        if "[TOOL_CALL]" in response_upper or "GAME ACTION:" in response_upper or "TEST RESULT:" in response_upper:
            return "tool_result"
        elif "[EVALUATION]" in response_upper or "DIAGNOSIS EVALUATION:" in response_upper:
            return "evaluation"
        else:
            # Default: patient dialogue
            return "patient_dialogue"

    def _extract_metadata(self, response: str, response_type: str) -> Dict:
        """
        Extract metadata from response based on type.

        Args:
            response: Raw response text
            response_type: Type of response

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}

        if response_type == "tool_result":
            # Extract cost information
            cost_match = re.search(r'COST:\s*\$?(\d+)\s*resource', response, re.IGNORECASE)
            if cost_match:
                metadata["cost"] = int(cost_match.group(1))
            else:
                # Try alternative patterns
                cost_match = re.search(r'\$(\d+)', response)
                if cost_match:
                    metadata["cost"] = int(cost_match.group(1))

        elif response_type == "evaluation":
            # Extract diagnosis correctness
            if "CORRECT" in response.upper() or "✅" in response:
                metadata["correct"] = True
                metadata["score_change"] = 10  # Default positive score
            elif "INCORRECT" in response.upper() or "❌" in response:
                metadata["correct"] = False
                metadata["score_change"] = -10  # Default negative score

            # Try to extract specific score change
            score_match = re.search(r'score[:\s]*([+-]?\d+)', response, re.IGNORECASE)
            if score_match:
                metadata["score_change"] = int(score_match.group(1))

        return metadata

    def set_model(self, model: str):
        """
        Change World LLM model.

        Args:
            model: New model identifier
        """
        self.model = model

    def get_model(self) -> str:
        """Get current model"""
        return self.model

    def test_connection(self) -> bool:
        """
        Test API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.generate_response(
                messages=[{"role": "user", "content": "Hello"}],
                system_prompt="You are a test assistant. Respond with 'OK'.",
                max_tokens=10,
                use_unified_response=False
            )
            return len(response.content) > 0
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


# Recommended models for different use cases
RECOMMENDED_MODELS = {
    "claude-sonnet": {
        "id": "anthropic/claude-3.5-sonnet",
        "description": "Best for complex reasoning and realistic simulations",
        "cost": "$$",
    },
    "claude-haiku": {
        "id": "anthropic/claude-3-5-haiku",
        "description": "Fast and affordable, good for simple scenarios",
        "cost": "$",
    },
    "gpt-4o": {
        "id": "openai/gpt-4o",
        "description": "Excellent reasoning, good alternative to Claude",
        "cost": "$$",
    },
    "gemini-flash": {
        "id": "google/gemini-2.0-flash",
        "description": "Very fast, low cost, decent quality",
        "cost": "$",
    },
    "llama-3.3": {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "description": "Open source, good quality, moderate cost",
        "cost": "$",
    }
}


def get_recommended_models() -> Dict[str, Dict]:
    """Get list of recommended models with descriptions"""
    return RECOMMENDED_MODELS
