"""Tests for configuration management"""

import pytest
import os
from pathlib import Path

from msb.core.config import Config, ModelConfig, EvaluationConfig, MetricConfig


class TestConfig:
    """Test configuration classes"""
    
    def test_model_config_initialization(self):
        """Test ModelConfig initialization"""
        config = ModelConfig(
            api_key="test_key",
            model_name="test_model",
            max_tokens=500,
            temperature=0.5
        )
        
        assert config.api_key == "test_key"
        assert config.model_name == "test_model"
        assert config.max_tokens == 500
        assert config.temperature == 0.5
    
    def test_model_config_env_var(self):
        """Test ModelConfig API key from environment"""
        os.environ["TEST_MODEL_API_KEY"] = "env_api_key"
        
        config = ModelConfig(model_name="test_model")
        
        assert config.api_key == "env_api_key"
        
        # Cleanup
        del os.environ["TEST_MODEL_API_KEY"]
    
    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values"""
        config = EvaluationConfig()
        
        assert config.batch_size == 32
        assert config.languages == ["en"]
        assert config.metrics == ["safety", "factuality", "cultural"]
        assert config.save_intermediate == True
        assert config.output_dir == "results"
    
    def test_metric_config_defaults(self):
        """Test MetricConfig default values"""
        config = MetricConfig()
        
        assert config.safety["toxicity_threshold"] == 0.7
        assert "violence" in config.safety["harm_categories"]
        assert config.factuality["confidence_threshold"] == 0.8
        assert config.cultural["sensitivity_threshold"] == 0.8
    
    def test_config_load_from_yaml(self, sample_config_yaml):
        """Test loading configuration from YAML file"""
        config = Config(str(sample_config_yaml))
        
        assert "openai" in config.models
        assert config.models["openai"].api_key == "test_key"
        assert config.evaluation.batch_size == 16
        assert config.evaluation.languages == ["en"]
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = Config()
        config.models["test"] = ModelConfig(api_key="key", model_name="test")
        config.evaluation.batch_size = 10
        config.evaluation.languages = ["en"]
        config.evaluation.metrics = ["safety"]
        
        assert config.validate() == True
        
        # Invalid config - no models
        config2 = Config()
        assert config2.validate() == False
        
        # Invalid config - no languages
        config3 = Config()
        config3.models["test"] = ModelConfig(api_key="key", model_name="test")
        config3.evaluation.languages = []
        assert config3.validate() == False
    
    def test_get_model_config(self):
        """Test getting model configuration"""
        config = Config()
        config.models["openai"] = ModelConfig(
            api_key="key",
            model_name="gpt-4",
            max_tokens=1000
        )
        
        # Exact match
        model_config = config.get_model_config("openai")
        assert model_config is not None
        assert model_config.model_name == "gpt-4"
        
        # Partial match
        model_config2 = config.get_model_config("gpt-4-turbo")
        assert model_config2 is not None
        assert model_config2.api_key == "key"
        
        # No match
        model_config3 = config.get_model_config("unknown")
        assert model_config3 is None
    
    def test_config_save(self, temp_dir):
        """Test saving configuration"""
        config = Config()
        config.models["test"] = ModelConfig(
            api_key="key",
            model_name="test_model",
            max_tokens=100
        )
        config.evaluation.languages = ["en", "zh"]
        
        output_path = temp_dir / "saved_config.yaml"
        config.save(str(output_path))
        
        assert output_path.exists()
        
        # Load saved config
        loaded_config = Config(str(output_path))
        assert "test" in loaded_config.models
        assert loaded_config.evaluation.languages == ["en", "zh"]