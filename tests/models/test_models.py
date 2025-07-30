"""Tests for model interfaces"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from msb.models import get_model, BaseModel
from msb.models.openai import OpenAIModel
from msb.models.anthropic import AnthropicModel
from msb.models.cohere import CohereModel


class TestBaseModel:
    """Test base model functionality"""
    
    def test_base_model_initialization(self):
        """Test BaseModel initialization"""
        config = {
            "api_key": "test_key",
            "max_tokens": 500,
            "temperature": 0.5,
            "timeout": 30,
            "retry_attempts": 3
        }
        
        # Create a concrete implementation for testing
        class TestModel(BaseModel):
            def generate(self, prompt, **kwargs):
                return "Test response"
        
        model = TestModel("test_model", config)
        
        assert model.model_name == "test_model"
        assert model.api_key == "test_key"
        assert model.max_tokens == 500
        assert model.temperature == 0.5
    
    def test_generate_batch(self):
        """Test batch generation"""
        class TestModel(BaseModel):
            def generate(self, prompt, **kwargs):
                return f"Response to: {prompt}"
        
        model = TestModel("test_model", {"api_key": "key"})
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = model.generate_batch(prompts, max_workers=2)
        
        assert len(responses) == 3
        assert all(r is not None for r in responses)
        assert responses[0] == "Response to: Prompt 1"
    
    def test_generate_with_retry(self):
        """Test generation with retry logic"""
        class TestModel(BaseModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.attempt_count = 0
            
            def generate(self, prompt, **kwargs):
                self.attempt_count += 1
                if self.attempt_count < 2:
                    raise Exception("Temporary error")
                return "Success"
        
        model = TestModel("test_model", {"api_key": "key", "retry_attempts": 3})
        
        result = model._generate_with_retry("Test prompt")
        assert result == "Success"
        assert model.attempt_count == 2
    
    def test_estimate_tokens(self):
        """Test token estimation"""
        model = BaseModel("test", {})
        
        # Test estimation
        text = "This is a test sentence with multiple words."
        estimated = model.estimate_tokens(text)
        
        # Rough estimate should be around text length / 4
        assert estimated > 0
        assert estimated < len(text)
    
    def test_truncate_prompt(self):
        """Test prompt truncation"""
        model = BaseModel("test", {})
        
        # Test no truncation needed
        short_prompt = "Short prompt"
        result = model.truncate_prompt(short_prompt, 100)
        assert result == short_prompt
        
        # Test truncation
        long_prompt = "This is a very long prompt. " * 100
        result = model.truncate_prompt(long_prompt, 50)
        assert len(result) < len(long_prompt)
        assert result.endswith("...")


class TestOpenAIModel:
    """Test OpenAI model interface"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_initialization_with_env_var(self):
        """Test initialization with environment variable"""
        with patch('msb.models.openai.OpenAI'):
            model = OpenAIModel("gpt-4", {})
            assert model.api_key == "test_key"
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key"""
        with pytest.raises(ValueError, match="API key not provided"):
            OpenAIModel("gpt-4", {})
    
    @patch('msb.models.openai.OpenAI')
    def test_generate(self, mock_openai_class):
        """Test generate method"""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        model = OpenAIModel("gpt-4", {"api_key": "test_key"})
        
        response = model.generate("Test prompt", max_tokens=100)
        
        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()
    
    def test_get_available_models(self):
        """Test getting available models"""
        with patch('msb.models.openai.OpenAI'):
            model = OpenAIModel("gpt-4", {"api_key": "test_key"})
            models = model.get_available_models()
            
            assert "gpt-4" in models
            assert "gpt-3.5-turbo" in models


class TestAnthropicModel:
    """Test Anthropic model interface"""
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    def test_initialization_with_env_var(self):
        """Test initialization with environment variable"""
        with patch('msb.models.anthropic.Anthropic'):
            model = AnthropicModel("claude-3", {})
            assert model.api_key == "test_key"
    
    @patch('msb.models.anthropic.Anthropic')
    def test_generate(self, mock_anthropic_class):
        """Test generate method"""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        
        model = AnthropicModel("claude-3", {"api_key": "test_key"})
        
        response = model.generate("Test prompt")
        
        assert response == "Test response"
        mock_client.messages.create.assert_called_once()


class TestModelFactory:
    """Test model factory functions"""
    
    def test_get_model_by_name(self):
        """Test getting model by name"""
        with patch('msb.models.openai.OpenAI'):
            model = get_model("gpt-4", {"openai": {"api_key": "test_key"}})
            assert isinstance(model, OpenAIModel)
        
        with patch('msb.models.anthropic.Anthropic'):
            model = get_model("claude-3", {"anthropic": {"api_key": "test_key"}})
            assert isinstance(model, AnthropicModel)
    
    def test_get_model_invalid(self):
        """Test getting invalid model"""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model("unknown_model", {})
    
    def test_get_model_instance(self):
        """Test passing model instance"""
        mock_model = Mock(spec=BaseModel)
        result = get_model(mock_model, {})
        assert result is mock_model