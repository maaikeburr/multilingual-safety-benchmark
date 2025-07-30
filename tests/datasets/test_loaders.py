"""Tests for dataset loaders"""

import pytest
import json
from pathlib import Path

from msb.datasets import (
    get_dataset,
    list_available_datasets,
    validate_dataset,
    MultilingualSafetyDataset,
    CustomDataset,
    BaseDataset
)


class TestDatasetLoaders:
    """Test dataset loading functionality"""
    
    def test_get_dataset_builtin(self):
        """Test loading built-in dataset"""
        dataset = get_dataset("multilingual_safety")
        
        assert isinstance(dataset, MultilingualSafetyDataset)
        assert dataset.name == "multilingual_safety"
        assert len(dataset.get_languages()) > 0
    
    def test_get_dataset_from_file(self, sample_dataset_json):
        """Test loading dataset from file"""
        dataset = get_dataset(str(sample_dataset_json))
        
        assert isinstance(dataset, CustomDataset)
        assert "en" in dataset.get_languages()
        assert "zh" in dataset.get_languages()
    
    def test_get_dataset_invalid(self):
        """Test loading invalid dataset"""
        with pytest.raises(ValueError):
            get_dataset("nonexistent_dataset")
    
    def test_list_available_datasets(self):
        """Test listing available datasets"""
        datasets = list_available_datasets()
        
        assert "multilingual_safety" in datasets
        assert "description" in datasets["multilingual_safety"]
        assert "languages" in datasets["multilingual_safety"]
    
    def test_validate_dataset_valid(self):
        """Test validating a valid dataset"""
        dataset = MultilingualSafetyDataset()
        assert validate_dataset(dataset) == True
    
    def test_validate_dataset_invalid(self):
        """Test validating an invalid dataset"""
        # Create a mock invalid dataset
        class InvalidDataset:
            pass
        
        dataset = InvalidDataset()
        assert validate_dataset(dataset) == False


class TestMultilingualSafetyDataset:
    """Test the built-in multilingual safety dataset"""
    
    def test_initialization(self):
        """Test dataset initialization"""
        dataset = MultilingualSafetyDataset()
        
        assert dataset.name == "multilingual_safety"
        assert len(dataset.get_languages()) >= 5
        assert dataset.metadata["total_samples"] > 0
    
    def test_get_samples(self):
        """Test getting samples"""
        dataset = MultilingualSafetyDataset()
        
        # Get all English samples
        en_samples = dataset.get_samples("en")
        assert len(en_samples) > 0
        assert all("prompt" in sample for sample in en_samples)
        
        # Get limited samples
        limited_samples = dataset.get_samples("en", max_samples=2)
        assert len(limited_samples) == 2
    
    def test_get_samples_invalid_language(self):
        """Test getting samples for invalid language"""
        dataset = MultilingualSafetyDataset()
        
        samples = dataset.get_samples("invalid_lang")
        assert samples == []
    
    def test_get_statistics(self):
        """Test getting dataset statistics"""
        dataset = MultilingualSafetyDataset()
        stats = dataset.get_statistics()
        
        assert "total_samples" in stats
        assert "languages" in stats
        assert "categories" in stats
        assert stats["total_samples"] > 0


class TestCustomDataset:
    """Test custom dataset loading"""
    
    def test_load_json_dataset(self, temp_dir):
        """Test loading JSON dataset"""
        # Create test dataset
        dataset_path = temp_dir / "test.json"
        data = {
            "en": [
                {"id": "1", "prompt": "Test 1", "category": "test"},
                {"id": "2", "prompt": "Test 2", "category": "test"}
            ],
            "fr": [
                {"id": "3", "prompt": "Test 3", "category": "test"}
            ]
        }
        dataset_path.write_text(json.dumps(data))
        
        # Load dataset
        dataset = CustomDataset(str(dataset_path), format="json")
        
        assert dataset.get_languages() == ["en", "fr"]
        assert len(dataset.get_samples("en")) == 2
        assert len(dataset.get_samples("fr")) == 1
    
    def test_load_csv_dataset(self, temp_dir):
        """Test loading CSV dataset"""
        import pandas as pd
        
        # Create test CSV
        dataset_path = temp_dir / "test.csv"
        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "prompt": ["Test 1", "Test 2", "Test 3"],
            "language": ["en", "en", "zh"],
            "category": ["test", "test", "test"]
        })
        df.to_csv(dataset_path, index=False)
        
        # Load dataset
        dataset = CustomDataset(str(dataset_path), format="csv")
        
        assert "en" in dataset.get_languages()
        assert "zh" in dataset.get_languages()
        assert len(dataset.get_samples("en")) == 2
    
    def test_load_jsonl_dataset(self, temp_dir):
        """Test loading JSONL dataset"""
        # Create test JSONL
        dataset_path = temp_dir / "test.jsonl"
        lines = [
            json.dumps({"id": "1", "prompt": "Test 1", "language": "en"}),
            json.dumps({"id": "2", "prompt": "Test 2", "language": "en"}),
            json.dumps({"id": "3", "prompt": "Test 3", "language": "fr"})
        ]
        dataset_path.write_text("\n".join(lines))
        
        # Load dataset
        dataset = CustomDataset(str(dataset_path), format="jsonl")
        
        assert "en" in dataset.get_languages()
        assert "fr" in dataset.get_languages()
    
    def test_filter_by_category(self):
        """Test filtering samples by category"""
        dataset = MultilingualSafetyDataset()
        
        harmful_samples = dataset.filter_by_category("harmful_advice")
        if harmful_samples:  # Only test if category exists
            assert all(s["category"] == "harmful_advice" for s in harmful_samples)