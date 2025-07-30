"""Main evaluation engine for MSB"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

from .config import Config
from ..models import get_model
from ..datasets import get_dataset
from ..evaluation import SafetyEvaluator, FactualityEvaluator, CulturalEvaluator
from ..utils.logger import setup_logger
from ..utils.reporter import ReportGenerator

logger = setup_logger(__name__)


class MSBEvaluator:
    """Main evaluator class for Multilingual Safety Benchmark"""
    
    def __init__(self, config: Optional[Union[str, Config]] = None):
        """
        Initialize the evaluator
        
        Args:
            config: Path to config file or Config object
        """
        if isinstance(config, str):
            self.config = Config(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = Config()
        
        # Initialize evaluators
        self.evaluators = {
            "safety": SafetyEvaluator(self.config.metrics.safety),
            "factuality": FactualityEvaluator(self.config.metrics.factuality),
            "cultural": CulturalEvaluator(self.config.metrics.cultural)
        }
        
        # Results storage
        self.results = {}
        self.metadata = {
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "config": self.config.raw_config
        }
    
    def evaluate(
        self,
        model: str,
        dataset: str,
        languages: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation on a model with a dataset
        
        Args:
            model: Model name or identifier
            dataset: Dataset name or path
            languages: List of languages to evaluate (None = use config)
            metrics: List of metrics to compute (None = use config)
            max_samples: Maximum samples per language (None = use all)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing evaluation results
        """
        # Use config defaults if not specified
        languages = languages or self.config.evaluation.languages
        metrics = metrics or self.config.evaluation.metrics
        max_samples = max_samples or self.config.evaluation.max_samples
        
        logger.info(f"Starting evaluation: model={model}, dataset={dataset}")
        logger.info(f"Languages: {languages}, Metrics: {metrics}")
        
        # Load model
        try:
            model_instance = get_model(model, self.config)
        except Exception as e:
            logger.error(f"Failed to load model {model}: {e}")
            raise
        
        # Load dataset
        try:
            dataset_instance = get_dataset(dataset)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset}: {e}")
            raise
        
        # Initialize results structure
        results = {
            "model": model,
            "dataset": dataset,
            "languages": {},
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "languages_evaluated": languages,
                "metrics_computed": metrics,
                "total_samples": 0
            }
        }
        
        # Evaluate each language
        for language in languages:
            logger.info(f"Evaluating language: {language}")
            results["languages"][language] = self._evaluate_language(
                model_instance,
                dataset_instance,
                language,
                metrics,
                max_samples
            )
        
        # Compute aggregate metrics
        results["aggregate"] = self._compute_aggregate_metrics(results["languages"])
        
        # Update metadata
        results["metadata"]["end_time"] = datetime.now().isoformat()
        results["metadata"]["total_samples"] = sum(
            lang_results.get("num_samples", 0) 
            for lang_results in results["languages"].values()
        )
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        self.results = results
        return results
    
    def _evaluate_language(
        self,
        model,
        dataset,
        language: str,
        metrics: List[str],
        max_samples: Optional[int]
    ) -> Dict[str, Any]:
        """Evaluate a single language"""
        # Get samples for this language
        samples = dataset.get_samples(language, max_samples)
        
        if not samples:
            logger.warning(f"No samples found for language: {language}")
            return {"error": "No samples found", "num_samples": 0}
        
        logger.info(f"Evaluating {len(samples)} samples for {language}")
        
        # Initialize results
        language_results = {
            "num_samples": len(samples),
            "metrics": {metric: [] for metric in metrics},
            "errors": []
        }
        
        # Process samples in batches
        batch_size = self.config.evaluation.batch_size
        
        with tqdm(total=len(samples), desc=f"Processing {language}") as pbar:
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                batch_results = self._process_batch(model, batch, language, metrics)
                
                # Aggregate batch results
                for metric in metrics:
                    if metric in batch_results:
                        language_results["metrics"][metric].extend(batch_results[metric])
                
                if "errors" in batch_results:
                    language_results["errors"].extend(batch_results["errors"])
                
                pbar.update(len(batch))
        
        # Compute summary statistics
        for metric in metrics:
            if language_results["metrics"][metric]:
                language_results[f"{metric}_summary"] = self._compute_summary_stats(
                    language_results["metrics"][metric]
                )
        
        return language_results
    
    def _process_batch(
        self,
        model,
        batch: List[Dict],
        language: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Process a batch of samples"""
        results = {metric: [] for metric in metrics}
        results["errors"] = []
        
        # Generate model responses
        prompts = [sample.get("prompt", sample.get("text", "")) for sample in batch]
        
        try:
            responses = model.generate_batch(prompts, language=language)
        except Exception as e:
            logger.error(f"Error generating responses: {e}")
            results["errors"].append({"error": str(e), "batch_size": len(batch)})
            return results
        
        # Evaluate each response
        for i, (sample, response) in enumerate(zip(batch, responses)):
            if response is None:
                results["errors"].append({"sample_id": sample.get("id", i), "error": "No response"})
                continue
            
            # Run each metric evaluator
            for metric in metrics:
                if metric in self.evaluators:
                    try:
                        score = self.evaluators[metric].evaluate(
                            prompt=sample.get("prompt", sample.get("text", "")),
                            response=response,
                            context=sample.get("context", {}),
                            language=language
                        )
                        results[metric].append(score)
                    except Exception as e:
                        logger.error(f"Error computing {metric} for sample {i}: {e}")
                        results["errors"].append({
                            "sample_id": sample.get("id", i),
                            "metric": metric,
                            "error": str(e)
                        })
        
        return results
    
    def _compute_summary_stats(self, scores: List[Dict]) -> Dict[str, float]:
        """Compute summary statistics for a list of scores"""
        if not scores:
            return {}
        
        # Extract numeric scores
        numeric_scores = []
        for score in scores:
            if isinstance(score, dict) and "score" in score:
                numeric_scores.append(score["score"])
            elif isinstance(score, (int, float)):
                numeric_scores.append(score)
        
        if not numeric_scores:
            return {}
        
        df = pd.DataFrame({"score": numeric_scores})
        
        return {
            "mean": float(df["score"].mean()),
            "std": float(df["score"].std()),
            "min": float(df["score"].min()),
            "max": float(df["score"].max()),
            "median": float(df["score"].median()),
            "count": len(numeric_scores)
        }
    
    def _compute_aggregate_metrics(self, language_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute aggregate metrics across all languages"""
        aggregate = {
            "total_samples": sum(
                lang.get("num_samples", 0) for lang in language_results.values()
            ),
            "total_errors": sum(
                len(lang.get("errors", [])) for lang in language_results.values()
            ),
            "metrics": {}
        }
        
        # Aggregate each metric
        all_metrics = set()
        for lang_data in language_results.values():
            if isinstance(lang_data, dict) and "metrics" in lang_data:
                all_metrics.update(lang_data["metrics"].keys())
        
        for metric in all_metrics:
            all_scores = []
            for lang_data in language_results.values():
                if isinstance(lang_data, dict) and "metrics" in lang_data:
                    if metric in lang_data["metrics"]:
                        all_scores.extend(lang_data["metrics"][metric])
            
            if all_scores:
                aggregate["metrics"][metric] = self._compute_summary_stats(all_scores)
        
        return aggregate
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to disk"""
        output_dir = Path(self.config.evaluation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = results["model"].replace("/", "_").replace(":", "_")
        filename = f"evaluation_{model_name}_{timestamp}.json"
        
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(
        self,
        results: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        format: str = "html"
    ) -> str:
        """
        Generate evaluation report
        
        Args:
            results: Results dictionary (uses self.results if None)
            output_dir: Output directory for report
            format: Report format (html, pdf, markdown)
            
        Returns:
            Path to generated report
        """
        results = results or self.results
        if not results:
            raise ValueError("No results to generate report from")
        
        output_dir = output_dir or self.config.evaluation.output_dir
        
        reporter = ReportGenerator(self.config)
        report_path = reporter.generate(results, output_dir, format)
        
        logger.info(f"Report generated: {report_path}")
        return report_path
    
    def compare_models(
        self,
        models: List[str],
        dataset: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same dataset
        
        Args:
            models: List of model names
            dataset: Dataset to use
            **kwargs: Additional arguments passed to evaluate()
            
        Returns:
            Comparison results
        """
        comparison_results = {
            "models": models,
            "dataset": dataset,
            "individual_results": {},
            "comparison": {}
        }
        
        # Evaluate each model
        for model in models:
            logger.info(f"Evaluating model: {model}")
            try:
                results = self.evaluate(model, dataset, save_results=False, **kwargs)
                comparison_results["individual_results"][model] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model}: {e}")
                comparison_results["individual_results"][model] = {"error": str(e)}
        
        # Generate comparison metrics
        comparison_results["comparison"] = self._generate_comparison(
            comparison_results["individual_results"]
        )
        
        return comparison_results
    
    def _generate_comparison(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comparison metrics between models"""
        comparison = {"rankings": {}, "details": {}}
        
        # Extract valid results
        valid_results = {
            model: results for model, results in model_results.items()
            if "error" not in results and "aggregate" in results
        }
        
        if not valid_results:
            return comparison
        
        # Compare each metric
        metrics = set()
        for results in valid_results.values():
            if "aggregate" in results and "metrics" in results["aggregate"]:
                metrics.update(results["aggregate"]["metrics"].keys())
        
        for metric in metrics:
            metric_scores = {}
            for model, results in valid_results.items():
                if metric in results["aggregate"]["metrics"]:
                    metric_scores[model] = results["aggregate"]["metrics"][metric].get("mean", 0)
            
            # Rank models by metric
            if metric_scores:
                sorted_models = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
                comparison["rankings"][metric] = [model for model, _ in sorted_models]
                comparison["details"][metric] = metric_scores
        
        return comparison