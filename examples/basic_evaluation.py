"""Example: Basic evaluation script"""

import os
from msb import MSBEvaluator

def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize evaluator
    print("Initializing MSB evaluator...")
    evaluator = MSBEvaluator("configs/minimal.yaml")
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate(
        model="gpt-3.5-turbo",
        dataset="multilingual_safety",
        languages=["en"],  # Just English for quick test
        max_samples=5      # Only 5 samples
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Model: {results['model']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Total samples: {results['metadata']['total_samples']}")
    
    # Print metrics
    if "aggregate" in results and "metrics" in results["aggregate"]:
        print("\nMetrics:")
        for metric, stats in results["aggregate"]["metrics"].items():
            if isinstance(stats, dict) and "mean" in stats:
                print(f"  {metric}: {stats['mean']:.3f}")
    
    # Print any errors
    total_errors = results.get("aggregate", {}).get("total_errors", 0)
    if total_errors > 0:
        print(f"\nErrors encountered: {total_errors}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()