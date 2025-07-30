"""Model loading utilities and factory"""

from typing import Union, Optional, Dict, Any
import logging

from .base import BaseModel
from .openai import OpenAIModel
from .anthropic import AnthropicModel
from .cohere import CohereModel

logger = logging.getLogger(__name__)

# Model registry
MODEL_REGISTRY = {
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
    "cohere": CohereModel,
    "gpt": OpenAIModel,  # Aliases
    "claude": AnthropicModel,
    "command": CohereModel
}


def get_model(
    model: Union[str, BaseModel],
    config: Optional[Any] = None
) -> BaseModel:
    """
    Get a model instance
    
    Args:
        model: Model name, identifier, or instance
        config: Configuration object or dict
        
    Returns:
        Model instance
    """
    if isinstance(model, BaseModel):
        return model
    
    if isinstance(model, str):
        model_lower = model.lower()
        
        # Extract model config if config object provided
        model_config = None
        if config:
            if hasattr(config, 'get_model_config'):
                model_config = config.get_model_config(model)
                if model_config:
                    model_config = {
                        "api_key": model_config.api_key,
                        "max_tokens": model_config.max_tokens,
                        "temperature": model_config.temperature,
                        "timeout": model_config.timeout,
                        "retry_attempts": model_config.retry_attempts
                    }
            elif isinstance(config, dict):
                model_config = config
        
        # Check if it's a known model type
        for key, model_class in MODEL_REGISTRY.items():
            if key in model_lower:
                return model_class(model, model_config)
        
        # Check specific model names
        if "gpt" in model_lower:
            return OpenAIModel(model, model_config)
        elif "claude" in model_lower:
            return AnthropicModel(model, model_config)
        elif "command" in model_lower:
            return CohereModel(model, model_config)
        else:
            raise ValueError(f"Unknown model type: {model}")
    
    raise TypeError(f"Invalid model type: {type(model)}")


def list_available_models() -> Dict[str, list]:
    """List all available models by provider"""
    return {
        "openai": OpenAIModel("dummy").get_available_models(),
        "anthropic": AnthropicModel("dummy").get_available_models(),
        "cohere": CohereModel("dummy").get_available_models()
    }


def validate_model(model: BaseModel) -> bool:
    """
    Validate a model is properly configured
    
    Args:
        model: Model to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if API key is present
        if not model.validate_api_key():
            return False
        
        # Try a simple generation to test connectivity
        test_response = model.generate("Hello", max_tokens=10)
        if not test_response:
            logger.error("Model returned empty response")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False