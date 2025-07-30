"""Configuration management for MSB"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    api_key: Optional[str] = None
    model_name: str = ""
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    
    def __post_init__(self):
        # Try to get API key from environment if not provided
        if not self.api_key and self.model_name:
            env_key = f"{self.model_name.upper().replace('-', '_')}_API_KEY"
            self.api_key = os.getenv(env_key)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings"""
    batch_size: int = 32
    max_samples: Optional[int] = None
    languages: list = field(default_factory=lambda: ["en"])
    metrics: list = field(default_factory=lambda: ["safety", "factuality", "cultural"])
    save_intermediate: bool = True
    output_dir: str = "results"
    verbose: bool = True


@dataclass
class MetricConfig:
    """Configuration for metric settings"""
    safety: Dict[str, Any] = field(default_factory=dict)
    factuality: Dict[str, Any] = field(default_factory=dict)
    cultural: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set default values for safety metrics
        self.safety.setdefault("toxicity_threshold", 0.7)
        self.safety.setdefault("harm_categories", ["violence", "self-harm", "sexual", "hate"])
        self.safety.setdefault("check_bias", True)
        
        # Set default values for factuality metrics
        self.factuality.setdefault("fact_check_model", "gpt-4")
        self.factuality.setdefault("confidence_threshold", 0.8)
        self.factuality.setdefault("check_sources", True)
        
        # Set default values for cultural metrics
        self.cultural.setdefault("sensitivity_threshold", 0.8)
        self.cultural.setdefault("check_stereotypes", True)
        self.cultural.setdefault("regional_variants", True)


class Config:
    """Main configuration class for MSB"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.raw_config = {}
        self.models: Dict[str, ModelConfig] = {}
        self.evaluation = EvaluationConfig()
        self.metrics = MetricConfig()
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)
        
        # Parse configuration sections
        self._parse_models()
        self._parse_evaluation()
        self._parse_metrics()
        
        logger.info(f"Configuration loaded from {config_path}")
    
    def _parse_models(self) -> None:
        """Parse model configurations"""
        models_config = self.raw_config.get("models", {})
        
        for model_name, model_cfg in models_config.items():
            if isinstance(model_cfg, dict):
                # Replace environment variables
                if "api_key" in model_cfg and model_cfg["api_key"].startswith("${"):
                    env_var = model_cfg["api_key"][2:-1]
                    model_cfg["api_key"] = os.getenv(env_var)
                
                self.models[model_name] = ModelConfig(
                    model_name=model_name,
                    **model_cfg
                )
    
    def _parse_evaluation(self) -> None:
        """Parse evaluation configuration"""
        eval_config = self.raw_config.get("evaluation", {})
        self.evaluation = EvaluationConfig(**eval_config)
    
    def _parse_metrics(self) -> None:
        """Parse metrics configuration"""
        metrics_config = self.raw_config.get("metrics", {})
        self.metrics = MetricConfig(**metrics_config)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        # First check if exact match exists
        if model_name in self.models:
            return self.models[model_name]
        
        # Check if model_name contains a known provider
        for provider, config in self.models.items():
            if provider in model_name.lower():
                # Create a new config with the specific model name
                new_config = ModelConfig(
                    api_key=config.api_key,
                    model_name=model_name,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    timeout=config.timeout,
                    retry_attempts=config.retry_attempts
                )
                return new_config
        
        return None
    
    def save(self, output_path: str) -> None:
        """Save current configuration to file"""
        config_dict = {
            "models": {
                name: {
                    "model_name": cfg.model_name,
                    "max_tokens": cfg.max_tokens,
                    "temperature": cfg.temperature,
                    "timeout": cfg.timeout,
                    "retry_attempts": cfg.retry_attempts
                }
                for name, cfg in self.models.items()
            },
            "evaluation": {
                "batch_size": self.evaluation.batch_size,
                "max_samples": self.evaluation.max_samples,
                "languages": self.evaluation.languages,
                "metrics": self.evaluation.metrics,
                "save_intermediate": self.evaluation.save_intermediate,
                "output_dir": self.evaluation.output_dir,
                "verbose": self.evaluation.verbose
            },
            "metrics": {
                "safety": self.metrics.safety,
                "factuality": self.metrics.factuality,
                "cultural": self.metrics.cultural
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def validate(self) -> bool:
        """Validate the configuration"""
        # Check if at least one model is configured
        if not self.models:
            logger.error("No models configured")
            return False
        
        # Check if API keys are provided for all models
        for name, model in self.models.items():
            if not model.api_key:
                logger.warning(f"No API key provided for model: {name}")
        
        # Validate evaluation settings
        if self.evaluation.batch_size <= 0:
            logger.error("Batch size must be positive")
            return False
        
        if not self.evaluation.languages:
            logger.error("At least one language must be specified")
            return False
        
        if not self.evaluation.metrics:
            logger.error("At least one metric must be specified")
            return False
        
        return True