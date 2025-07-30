"""Tests for the core evaluator"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from msb.core.evaluator import MSBEvaluator
from msb.evaluation import SafetyScore, FactualityScore


class TestMSBEvaluator:
    """Test the main evaluator class"""
    
    def test_initialization(self, mock_config):
        """Test evaluator initialization"""
        evaluator = MSBEvaluator(mock_config)
        
        assert evaluator.config == mock_config
        assert "safety" in evaluator.evaluators
        assert "factuality" in evaluator.evaluators
    
    @patch('msb.core.evaluator.get_model')
    @patch('msb.core.evaluator.get_dataset')
    def test_evaluate_basic(self, mock_get_dataset, mock_get_model, 
                           mock_config, mock_model, mock_dataset):
        """Test basic evaluation flow"""
        # Setup mocks
        mock_get_model.return_value = mock_model
        mock_get_dataset.return_value = mock_dataset
        
        # Mock evaluators
        mock_safety_score = SafetyScore(
            toxicity=0.1,
            harm_probability=0.1,
            bias_score=0.9,
            overall_safety=0.9,
            details={}
        )
        
        mock_factuality_score = FactualityScore(
            accuracy=0.85,
            hallucination_rate=0.1,
            source_reliability=0.8,
            overall_factuality=0.85,
            details={}
        )
        
        evaluator = MSBEvaluator(mock_config)
        evaluator.evaluators["safety"].evaluate = Mock(return_value=mock_safety_score)
        evaluator.evaluators["factuality"].evaluate = Mock(return_value=mock_factuality_score)
        
        # Run evaluation
        results = evaluator.evaluate(
            model="test_model",
            dataset="test_dataset",
            languages=["en"],
            max_samples=2,
            save_results=False
        )
        
        # Verify results structure
        assert results["model"] == "test_model"
        assert results["dataset"] == "test_dataset"
        assert "languages" in results
        assert "en" in results["languages"]
        assert results["languages"]["en"]["num_samples"] == 2
        assert "aggregate" in results
        assert "metadata" in results
    
    def test_evaluate_with_errors(self, mock_config, mock_model, mock_dataset):
        """Test evaluation with errors"""
        with patch('msb.core.evaluator.get_model') as mock_get_model:
            mock_get_model.side_effect = Exception("Model loading failed")
            
            evaluator = MSBEvaluator(mock_config)
            
            with pytest.raises(Exception) as exc_info:
                evaluator.evaluate("test_model", "test_dataset")
            
            assert "Model loading failed" in str(exc_info.value)
    
    @patch('msb.core.evaluator.get_model')
    @patch('msb.core.evaluator.get_dataset')
    def test_compare_models(self, mock_get_dataset, mock_get_model,
                          mock_config, mock_model, mock_dataset):
        """Test model comparison"""
        mock_get_model.return_value = mock_model
        mock_get_dataset.return_value = mock_dataset
        
        evaluator = MSBEvaluator(mock_config)
        
        # Mock the evaluate method
        evaluator.evaluate = Mock(return_value={
            "model": "test_model",
            "aggregate": {
                "metrics": {
                    "safety": {"mean": 0.85},
                    "factuality": {"mean": 0.8}
                }
            }
        })
        
        # Compare models
        results = evaluator.compare_models(
            models=["model1", "model2"],
            dataset="test_dataset"
        )
        
        assert "models" in results
        assert results["models"] == ["model1", "model2"]
        assert "individual_results" in results
        assert "comparison" in results
    
    def test_generate_report(self, mock_config, sample_results, temp_dir):
        """Test report generation"""
        evaluator = MSBEvaluator(mock_config)
        evaluator.results = sample_results
        
        with patch('msb.utils.reporter.ReportGenerator.generate') as mock_generate:
            mock_generate.return_value = str(temp_dir / "report.html")
            
            report_path = evaluator.generate_report(output_dir=str(temp_dir))
            
            assert "report.html" in report_path
            mock_generate.assert_called_once()
    
    def test_compute_summary_stats(self, mock_config):
        """Test summary statistics computation"""
        evaluator = MSBEvaluator(mock_config)
        
        scores = [
            {"score": 0.8},
            {"score": 0.9},
            {"score": 0.7}
        ]
        
        stats = evaluator._compute_summary_stats(scores)
        
        assert stats["mean"] == pytest.approx(0.8, 0.01)
        assert stats["min"] == 0.7
        assert stats["max"] == 0.9
        assert stats["count"] == 3
    
    def test_aggregate_metrics(self, mock_config):
        """Test metric aggregation across languages"""
        evaluator = MSBEvaluator(mock_config)
        
        language_results = {
            "en": {
                "num_samples": 5,
                "metrics": {
                    "safety": [{"score": 0.8}, {"score": 0.9}]
                }
            },
            "zh": {
                "num_samples": 5,
                "metrics": {
                    "safety": [{"score": 0.7}, {"score": 0.85}]
                }
            }
        }
        
        aggregate = evaluator._compute_aggregate_metrics(language_results)
        
        assert aggregate["total_samples"] == 10
        assert "metrics" in aggregate
        assert "safety" in aggregate["metrics"]