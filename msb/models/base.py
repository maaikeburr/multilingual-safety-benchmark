"""Base model interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.api_key = self.config.get("api_key")
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = self.config.get("timeout", 30)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        
        # Initialize model-specific settings
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize model-specific settings"""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response for a single prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text
        """
        pass
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_workers: int = 5,
        **kwargs
    ) -> List[Optional[str]]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_workers: Maximum concurrent workers
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of generated texts (None for failed generations)
        """
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._generate_with_retry,
                    prompt,
                    max_tokens,
                    temperature,
                    **kwargs
                ): i
                for i, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Failed to generate response for prompt {idx}: {e}")
                    results[idx] = None
        
        return results
    
    def _generate_with_retry(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """Generate with retry logic"""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                return self.generate(prompt, max_tokens, temperature, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
        
        logger.error(f"All retry attempts failed: {last_error}")
        raise last_error
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }
    
    def validate_api_key(self) -> bool:
        """Validate API key is present"""
        if not self.api_key:
            logger.error(f"No API key configured for {self.model_name}")
            return False
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text
        
        This is a rough approximation - actual tokenization varies by model
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def truncate_prompt(self, prompt: str, max_prompt_tokens: int) -> str:
        """
        Truncate prompt to fit within token limit
        
        Args:
            prompt: Input prompt
            max_prompt_tokens: Maximum tokens for prompt
            
        Returns:
            Truncated prompt
        """
        estimated_tokens = self.estimate_tokens(prompt)
        
        if estimated_tokens <= max_prompt_tokens:
            return prompt
        
        # Truncate to approximate token limit
        char_limit = max_prompt_tokens * 4
        truncated = prompt[:char_limit]
        
        # Try to truncate at sentence boundary
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        if last_period > char_limit * 0.8:
            return truncated[:last_period + 1]
        elif last_newline > char_limit * 0.8:
            return truncated[:last_newline]
        else:
            return truncated + "..."