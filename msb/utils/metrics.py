"""Metrics calculation utilities"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from scipy import stats
import pandas as pd


@dataclass
class MetricStatistics:
    """Statistics for a metric"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    count: int
    confidence_interval: tuple  # 95% CI
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "count": self.count,
            "confidence_interval": self.confidence_interval
        }


def calculate_statistics(scores: List[float]) -> MetricStatistics:
    """
    Calculate comprehensive statistics for a list of scores
    
    Args:
        scores: List of numeric scores
        
    Returns:
        MetricStatistics object
    """
    if not scores:
        return MetricStatistics(
            mean=0.0, std=0.0, min=0.0, max=0.0,
            median=0.0, q1=0.0, q3=0.0, count=0,
            confidence_interval=(0.0, 0.0)
        )
    
    scores_array = np.array(scores)
    
    # Calculate confidence interval
    confidence = 0.95
    n = len(scores)
    mean = np.mean(scores_array)
    sem = stats.sem(scores_array)  # Standard error of mean
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
    
    return MetricStatistics(
        mean=float(np.mean(scores_array)),
        std=float(np.std(scores_array)),
        min=float(np.min(scores_array)),
        max=float(np.max(scores_array)),
        median=float(np.median(scores_array)),
        q1=float(np.percentile(scores_array, 25)),
        q3=float(np.percentile(scores_array, 75)),
        count=len(scores),
        confidence_interval=(float(ci[0]), float(ci[1]))
    )


def aggregate_metrics(
    metric_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, MetricStatistics]:
    """
    Aggregate metric results across multiple evaluations
    
    Args:
        metric_results: Dictionary of metric names to lists of results
        
    Returns:
        Dictionary of metric names to statistics
    """
    aggregated = {}
    
    for metric_name, results in metric_results.items():
        # Extract scores from results
        scores = []
        for result in results:
            if isinstance(result, dict) and "score" in result:
                scores.append(result["score"])
            elif isinstance(result, (int, float)):
                scores.append(result)
        
        if scores:
            aggregated[metric_name] = calculate_statistics(scores)
    
    return aggregated


def compare_distributions(
    scores1: List[float],
    scores2: List[float],
    test: str = "mann-whitney"
) -> Dict[str, Any]:
    """
    Compare two score distributions
    
    Args:
        scores1: First set of scores
        scores2: Second set of scores
        test: Statistical test to use ("t-test", "mann-whitney", "ks")
        
    Returns:
        Dictionary with test results
    """
    if not scores1 or not scores2:
        return {"error": "Empty score lists provided"}
    
    result = {
        "test": test,
        "scores1_stats": calculate_statistics(scores1).to_dict(),
        "scores2_stats": calculate_statistics(scores2).to_dict()
    }
    
    if test == "t-test":
        # Independent samples t-test
        statistic, p_value = stats.ttest_ind(scores1, scores2)
        result["statistic"] = float(statistic)
        result["p_value"] = float(p_value)
        result["significant"] = p_value < 0.05
        
    elif test == "mann-whitney":
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
        result["statistic"] = float(statistic)
        result["p_value"] = float(p_value)
        result["significant"] = p_value < 0.05
        
    elif test == "ks":
        # Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(scores1, scores2)
        result["statistic"] = float(statistic)
        result["p_value"] = float(p_value)
        result["significant"] = p_value < 0.05
    
    # Effect size (Cohen's d)
    mean1, std1 = np.mean(scores1), np.std(scores1)
    mean2, std2 = np.mean(scores2), np.std(scores2)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    
    if pooled_std > 0:
        cohens_d = (mean1 - mean2) / pooled_std
        result["effect_size"] = float(cohens_d)
        result["effect_magnitude"] = interpret_effect_size(cohens_d)
    
    return result


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def calculate_correlation(
    scores1: List[float],
    scores2: List[float],
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Calculate correlation between two sets of scores
    
    Args:
        scores1: First set of scores
        scores2: Second set of scores
        method: Correlation method ("pearson", "spearman", "kendall")
        
    Returns:
        Dictionary with correlation results
    """
    if len(scores1) != len(scores2):
        return {"error": "Score lists must have same length"}
    
    if len(scores1) < 3:
        return {"error": "Need at least 3 pairs of scores"}
    
    if method == "pearson":
        corr, p_value = stats.pearsonr(scores1, scores2)
    elif method == "spearman":
        corr, p_value = stats.spearmanr(scores1, scores2)
    elif method == "kendall":
        corr, p_value = stats.kendalltau(scores1, scores2)
    else:
        return {"error": f"Unknown correlation method: {method}"}
    
    return {
        "method": method,
        "correlation": float(corr),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "strength": interpret_correlation_strength(corr)
    }


def interpret_correlation_strength(r: float) -> str:
    """Interpret correlation coefficient strength"""
    r = abs(r)
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "weak"
    elif r < 0.5:
        return "moderate"
    elif r < 0.7:
        return "strong"
    else:
        return "very strong"


def create_performance_matrix(
    results: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a performance matrix from evaluation results
    
    Args:
        results: Dictionary of model results
        metrics: List of metrics to include (None = all)
        
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    data = []
    
    for model, model_results in results.items():
        if "aggregate" not in model_results or "metrics" not in model_results["aggregate"]:
            continue
        
        row = {"model": model}
        
        for metric, stats in model_results["aggregate"]["metrics"].items():
            if metrics and metric not in metrics:
                continue
            
            if isinstance(stats, dict) and "mean" in stats:
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
            elif isinstance(stats, (int, float)):
                row[metric] = stats
        
        data.append(row)
    
    return pd.DataFrame(data).set_index("model")


def normalize_scores(
    scores: List[float],
    method: str = "min-max"
) -> List[float]:
    """
    Normalize scores to [0, 1] range
    
    Args:
        scores: List of scores to normalize
        method: Normalization method ("min-max", "z-score")
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    scores_array = np.array(scores)
    
    if method == "min-max":
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        
    elif method == "z-score":
        mean = np.mean(scores_array)
        std = np.std(scores_array)
        
        if std == 0:
            return [0.5] * len(scores)
        
        # Z-score normalization, then sigmoid to map to [0, 1]
        z_scores = (scores_array - mean) / std
        normalized = 1 / (1 + np.exp(-z_scores))
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.tolist()