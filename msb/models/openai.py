"""OpenAI model interface"""

import os
from typing import Optional, Dict, Any
import logging
from openai import OpenAI

from .base import BaseModel

logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    """OpenAI API model interface"""
    
    def _initialize(self) -> None:
        """Initialize OpenAI client"""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.validate_api_key():
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Set default model if not specified
        if not self.model_name or self.model_name == "openai":
            self.model_name = "gpt-4-turbo-preview"
        
        logger.info(f"Initialized OpenAI model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response using OpenAI API"""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.timeout,
                **kwargs
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_available_models(self) -> list:
        """Get list of available OpenAI models"""
        return [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-4-32k",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a generation"""
        # Pricing as of 2024 (prices per 1K tokens)
        pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-3.5-turbo-16k": {"input": 0.001, "output": 0.002}
        }
        
        if self.model_name in pricing:
            rates = pricing[self.model_name]
            input_cost = (input_tokens / 1000) * rates["input"]
            output_cost = (output_tokens / 1000) * rates["output"]
            return input_cost + output_cost
        
        return 0.0