"""Pytest configuration and shared fixtures"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from msb import Config, MSBEvaluator
from msb.models.base import BaseModel
from msb.datasets.base import BaseDataset


@pytest.fixture
def mock_config():
    """Create a mock configuration"""
    config = Config()
    config.models = {
        "test_model": {
            "api_key": "test_key",
            "model_name": "test_model",
            "max_tokens": 100,
            "temperature": 0.7
        }
    }
    config.evaluation.languages = ["en", "zh"]
    config.evaluation.metrics = ["safety", "factuality"]
    config.evaluation.batch_size = 2
    config.evaluation.output_dir = "test_results"
    return config


@pytest.fixture
def mock_model():
    """Create a mock model"""
    model = Mock(spec=BaseModel)
    model.model_name = "test_model"
    model.generate.return_value = "This is a safe response."
    model.generate_batch.return_value = [
        "Response 1",
        "Response 2"
    ]
    return model


@pytest.fixture
def mock_dataset():
    """Create a mock dataset"""
    dataset = Mock(spec=BaseDataset)
    dataset.name = "test_dataset"
    dataset.get_languages.return_value = ["en", "zh"]
    dataset.get_samples.side_effect = lambda lang, max_samples=None: [
        {
            "id": f"{lang}_001",
            "prompt": "Test prompt 1",
            "category": "educational",
            "expected_refusal": False
        },
        {
            "id": f"{lang}_002", 
            "prompt": "Harmful prompt",
            "category": "harmful",
            "expected_refusal": True
        }
    ][:max_samples] if max_samples else [
        {
            "id": f"{lang}_001",
            "prompt": "Test prompt 1",
            "category": "educational",
            "expected_refusal": False
        },
        {
            "id": f"{lang}_002",
            "prompt": "Harmful prompt", 
            "category": "harmful",
            "expected_refusal": True
        }
    ]
    return dataset


@pytest.fixture
def temp_dir():
    """Create a temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_results():
    """Create sample evaluation results"""
    return {
        "model": "test_model",
        "dataset": "test_dataset",
        "languages": {
            "en": {
                "num_samples": 10,
                "metrics": {
                    "safety": [{"score": 0.9}, {"score": 0.8}],
                    "factuality": [{"score": 0.85}, {"score": 0.75}]
                },
                "safety_summary": {
                    "mean": 0.85,
                    "std": 0.05,
                    "min": 0.8,
                    "max": 0.9,
                    "median": 0.85
                },
                "factuality_summary": {
                    "mean": 0.8,
                    "std": 0.05,
                    "min": 0.75,
                    "max": 0.85,
                    "median": 0.8
                }
            }
        },
        "aggregate": {
            "total_samples": 10,
            "metrics": {
                "safety": {
                    "mean": 0.85,
                    "std": 0.05
                },
                "factuality": {
                    "mean": 0.8,
                    "std": 0.05
                }
            }
        },
        "metadata": {
            "total_samples": 10,
            "languages_evaluated": ["en"]
        }
    }


@pytest.fixture
def sample_config_yaml(temp_dir):
    """Create a sample config YAML file"""
    config_path = temp_dir / "test_config.yaml"
    config_content = """
models:
  openai:
    api_key: test_key
    default_model: gpt-3.5-turbo
    max_tokens: 100

evaluation:
  batch_size: 16
  languages:
    - en
  metrics:
    - safety
  output_dir: test_results

metrics:
  safety:
    toxicity_threshold: 0.7
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_dataset_json(temp_dir):
    """Create a sample dataset JSON file"""
    dataset_path = temp_dir / "test_dataset.json"
    dataset_content = {
        "en": [
            {
                "id": "en_001",
                "prompt": "What is 2+2?",
                "category": "educational"
            }
        ],
        "zh": [
            {
                "id": "zh_001",
                "prompt": "2+2等于几？",
                "category": "educational"
            }
        ]
    }
    dataset_path.write_text(json.dumps(dataset_content))
    return dataset_path