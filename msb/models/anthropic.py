"""Anthropic model interface"""

import os
from typing import Optional, Dict, Any
import logging
from anthropic import Anthropic

from .base import BaseModel

logger = logging.getLogger(__name__)


class AnthropicModel(BaseModel):
    """Anthropic API model interface"""
    
    def _initialize(self) -> None:
        """Initialize Anthropic client"""
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.validate_api_key():
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=self.api_key)
        
        # Set default model if not specified
        if not self.model_name or self.model_name == "anthropic":
            self.model_name = "claude-3-opus-20240229"
        
        logger.info(f"Initialized Anthropic model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response using Anthropic API"""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        # Prepare the prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"Human: {prompt}\n\nAssistant:"
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt if system_prompt else None,
                **kwargs
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def get_available_models(self) -> list:
        """Get list of available Anthropic models"""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a generation"""
        # Pricing as of 2024 (prices per 1M tokens)
        pricing = {
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-2.1": {"input": 8.0, "output": 24.0},
            "claude-2.0": {"input": 8.0, "output": 24.0},
            "claude-instant-1.2": {"input": 0.8, "output": 2.4}
        }
        
        if self.model_name in pricing:
            rates = pricing[self.model_name]
            input_cost = (input_tokens / 1_000_000) * rates["input"]
            output_cost = (output_tokens / 1_000_000) * rates["output"]
            return input_cost + output_cost
        
        return 0.0