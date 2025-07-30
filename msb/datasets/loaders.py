"""Dataset loading utilities"""

from typing import Union, Optional
from pathlib import Path
import logging

from .base import BaseDataset, MultilingualSafetyDataset, HuggingFaceDataset, CustomDataset

logger = logging.getLogger(__name__)


def get_dataset(
    dataset: Union[str, BaseDataset],
    **kwargs
) -> BaseDataset:
    """
    Get a dataset instance
    
    Args:
        dataset: Dataset name, path, or instance
        **kwargs: Additional arguments for dataset initialization
        
    Returns:
        Dataset instance
    """
    if isinstance(dataset, BaseDataset):
        return dataset
    
    if isinstance(dataset, str):
        # Check if it's a file path
        path = Path(dataset)
        if path.exists():
            # Determine format from extension
            if path.suffix == ".json":
                return CustomDataset(dataset, format="json")
            elif path.suffix == ".csv":
                return CustomDataset(dataset, format="csv")
            elif path.suffix == ".jsonl":
                return CustomDataset(dataset, format="jsonl")
            else:
                logger.warning(f"Unknown file format: {path.suffix}, assuming JSON")
                return CustomDataset(dataset, format="json")
        
        # Check if it's a known dataset name
        dataset_lower = dataset.lower()
        
        if dataset_lower == "multilingual_safety":
            return MultilingualSafetyDataset(**kwargs)
        
        elif dataset_lower.startswith("hf:"):
            # HuggingFace dataset
            hf_name = dataset[3:]  # Remove "hf:" prefix
            if "/" in hf_name:
                # Has subset
                parts = hf_name.split("/", 1)
                return HuggingFaceDataset(parts[0], subset=parts[1])
            else:
                return HuggingFaceDataset(hf_name)
        
        else:
            # Try as HuggingFace dataset
            try:
                return HuggingFaceDataset(dataset, **kwargs)
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset}: {e}")
                raise ValueError(f"Unknown dataset: {dataset}")
    
    raise TypeError(f"Invalid dataset type: {type(dataset)}")


def list_available_datasets() -> dict:
    """List available built-in datasets"""
    return {
        "multilingual_safety": {
            "description": "Default multilingual safety evaluation dataset",
            "languages": ["en", "zh", "es", "ar", "hi"],
            "categories": ["harmful_advice", "educational", "illegal_activity", "privacy_violation", "health_advice"],
            "size": "small"
        }
    }


def validate_dataset(dataset: BaseDataset) -> bool:
    """
    Validate a dataset has required structure
    
    Args:
        dataset: Dataset to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check basic attributes
        if not hasattr(dataset, "get_samples"):
            logger.error("Dataset missing get_samples method")
            return False
        
        if not hasattr(dataset, "get_languages"):
            logger.error("Dataset missing get_languages method")
            return False
        
        # Check if dataset has data
        languages = dataset.get_languages()
        if not languages:
            logger.error("Dataset has no languages")
            return False
        
        # Check if we can get samples
        for lang in languages[:1]:  # Check first language only
            samples = dataset.get_samples(lang, max_samples=1)
            if not samples:
                logger.warning(f"No samples found for language {lang}")
            else:
                # Validate sample structure
                sample = samples[0]
                if "prompt" not in sample and "text" not in sample:
                    logger.error("Sample missing prompt/text field")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False