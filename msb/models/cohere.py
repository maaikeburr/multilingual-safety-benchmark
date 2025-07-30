"""Cohere model interface"""

import os
from typing import Optional, Dict, Any
import logging
import cohere

from .base import BaseModel

logger = logging.getLogger(__name__)


class CohereModel(BaseModel):
    """Cohere API model interface"""
    
    def _initialize(self) -> None:
        """Initialize Cohere client"""
        if not self.api_key:
            self.api_key = os.getenv("COHERE_API_KEY")
        
        if not self.validate_api_key():
            raise ValueError("Cohere API key not provided")
        
        self.client = cohere.Client(self.api_key)
        
        # Set default model if not specified
        if not self.model_name or self.model_name == "cohere":
            self.model_name = "command"
        
        logger.info(f"Initialized Cohere model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate response using Cohere API"""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.generations[0].text.strip()
            
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            raise
    
    def get_available_models(self) -> list:
        """Get list of available Cohere models"""
        return [
            "command",
            "command-light",
            "command-nightly",
            "command-light-nightly"
        ]
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a generation"""
        # Pricing as of 2024 (prices per 1M tokens)
        pricing = {
            "command": {"input": 0.4, "output": 0.8},
            "command-light": {"input": 0.15, "output": 0.6}
        }
        
        model_key = self.model_name
        if "nightly" in model_key:
            model_key = model_key.replace("-nightly", "")
        
        if model_key in pricing:
            rates = pricing[model_key]
            input_cost = (input_tokens / 1_000_000) * rates["input"]
            output_cost = (output_tokens / 1_000_000) * rates["output"]
            return input_cost + output_cost
        
        return 0.0