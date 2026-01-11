"""
Patient LLM - Separate LLM instance for patient simulation without privileged information
"""

import requests
from typing import List, Dict, Optional
import time


class PatientLLM:
    """
    OpenRouter-based Patient LLM for patient dialogue simulation.
    Separate from World LLM to maintain context isolation - patient does not have
    access to privileged information (test results, correct diagnosis, game state).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3-5-haiku",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    ):
        """
        Initialize Patient LLM.

        Args:
            api_key: OpenRouter API key
            model: Model identifier (default: Haiku for fast, affordable dialogue)
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        temperature: float = 0.8,  # Higher for natural variation
        max_tokens: int = 150,  # Shorter for patient responses
        retry_count: int = 3
    ) -> str:
        """
        Generate patient response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt for patient simulation (no privileged info)
            temperature: Sampling temperature (default 0.8 for natural dialogue)
            max_tokens: Maximum tokens to generate (default 150 for concise responses)
            retry_count: Number of retries on failure

        Returns:
            Generated patient response text

        Raises:
            RuntimeError: If API call fails after retries
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",  # Streamlit default
            "X-Title": "Activation Oracles Research Kit - Patient Simulation"
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
                return result["choices"][0]["message"]["content"]

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
                    raise RuntimeError(f"Patient LLM API error ({status_code}): {e.response.text}")

            except Exception as e:
                last_error = str(e)
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)

        # All retries failed
        raise RuntimeError(f"Patient LLM API call failed after {retry_count} attempts: {last_error}")

    def set_model(self, model: str):
        """
        Change Patient LLM model.

        Args:
            model: New model identifier
        """
        self.model = model

    def get_model(self) -> str:
        """Get current model"""
        return self.model

    def reset_conversation(self):
        """
        Reset conversation history (for scenarios that maintain history internally).
        This is a no-op for this implementation as conversation history is maintained
        by the scenario, not the LLM instance.
        """
        pass

    def test_connection(self) -> bool:
        """
        Test API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.generate_response(
                messages=[{"role": "user", "content": "Hello"}],
                system_prompt="You are a patient. Respond with a brief greeting.",
                max_tokens=10
            )
            return len(response) > 0
        except Exception as e:
            print(f"Patient LLM connection test failed: {e}")
            return False


# Recommended models for patient simulation (optimized for dialogue)
RECOMMENDED_PATIENT_MODELS = {
    "haiku": {
        "id": "anthropic/claude-3-5-haiku",
        "description": "Fast, natural dialogue, very affordable (Recommended)",
        "cost": "$",
    },
    "llama-small": {
        "id": "meta-llama/llama-3.2-3b-instruct",
        "description": "Open source, good for patient simulation",
        "cost": "$",
    },
    "gemini-flash": {
        "id": "google/gemini-2.0-flash",
        "description": "Very fast, low cost, decent quality",
        "cost": "$",
    },
    "qwen-small": {
        "id": "qwen/qwen-2.5-7b-instruct",
        "description": "Efficient, good for simple dialogue",
        "cost": "$",
    }
}


def get_recommended_patient_models() -> Dict[str, Dict]:
    """Get list of recommended models for patient simulation"""
    return RECOMMENDED_PATIENT_MODELS
