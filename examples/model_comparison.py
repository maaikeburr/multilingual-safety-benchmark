"""Example: Model comparison script"""

import os
from msb import MSBEvaluator
from msb.utils import create_performance_matrix
import pandas as pd

def main():
    # Check for API keys
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"Please set the following environment variables: {', '.join(missing_keys)}")
        return
    
    # Initialize evaluator
    print("Initializing MSB evaluator...")
    evaluator = MSBEvaluator("configs/default.yaml")
    
    # Models to compare
    models = [
        "gpt-3.5-turbo",
        "gpt-4",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229"
    ]
    
    # Run comparison
    print(f"\nComparing models: {', '.join(models)}")
    print("This may take a few minutes...")
    
    comparison_results = evaluator.compare_models(
        models=models,
        dataset="multilingual_safety",
        languages=["en", "zh"],  # Just 2 languages for speed
        max_samples=10          # Limited samples for demo
    )
    
    # Print comparison results
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Print rankings
    if "comparison" in comparison_results and "rankings" in comparison_results["comparison"]:
        rankings = comparison_results["comparison"]["rankings"]
        
        for metric, ranking in rankings.items():
            print(f"\n{metric.upper()} Rankings:")
            for i, model in enumerate(ranking, 1):
                score = comparison_results["comparison"]["details"][metric].get(model, 0)
                print(f"  {i}. {model}: {score:.3f}")
    
    # Create performance matrix
    print("\nPerformance Matrix:")
    print("-" * 60)
    
    # Extract individual results for matrix
    individual_results = comparison_results.get("individual_results", {})
    
    # Create a simple comparison table
    data = []
    for model, results in individual_results.items():
        if "error" not in results and "aggregate" in results:
            row = {"Model": model}
            for metric, stats in results["aggregate"]["metrics"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    row[metric] = f"{stats['mean']:.3f}"
            data.append(row)
    
    if data:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
    
    # Save detailed results
    output_file = "results/model_comparison.json"
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()