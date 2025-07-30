"""
Multilingual Safety Benchmark (MSB)
A comprehensive framework for evaluating LLM safety across languages.
"""

__version__ = "1.0.0"
__author__ = "MSB Contributors"

from .core.evaluator import MSBEvaluator
from .core.config import Config

__all__ = ["MSBEvaluator", "Config", "__version__"]