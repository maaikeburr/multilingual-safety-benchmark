# Quick Start Example

This example demonstrates how to use MSB to evaluate a language model.

## Prerequisites

1. Install MSB:
```bash
pip install -e .
```

2. Set up API keys:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## Basic Evaluation

```python
from msb import MSBEvaluator

# Initialize evaluator with default config
evaluator = MSBEvaluator("configs/default.yaml")

# Run evaluation on GPT-4
results = evaluator.evaluate(
    model="gpt-4",
    dataset="multilingual_safety",
    languages=["en", "zh", "es"],
    max_samples=10  # Use only 10 samples per language for quick test
)

# Generate HTML report
report_path = evaluator.generate_report(results)
print(f"Report saved to: {report_path}")

# Print summary statistics
print("\nEvaluation Summary:")
print(f"Model: {results['model']}")
print(f"Total samples: {results['metadata']['total_samples']}")

for metric, stats in results['aggregate']['metrics'].items():
    if isinstance(stats, dict):
        print(f"{metric}: {stats['mean']:.3f} (±{stats['std']:.3f})")
```

## Using Custom Dataset

```python
from msb import MSBEvaluator

# Load custom dataset
evaluator = MSBEvaluator("configs/default.yaml")

results = evaluator.evaluate(
    model="claude-3-opus-20240229",
    dataset="data/multilingual_safety_extended.json",
    languages=["en", "zh"]
)
```

## Command Line Usage

```bash
# Basic evaluation
msb evaluate --model gpt-4 --dataset multilingual_safety --languages en,zh,es

# With custom config
msb evaluate --config configs/production.yaml --model gpt-4

# Compare multiple models
msb compare --models gpt-4,claude-3-opus --dataset multilingual_safety

# Generate report from results
msb report --results results/evaluation_gpt-4_20240101.json --format html
```

## Output

The evaluation will produce:
1. JSON results file in `results/` directory
2. HTML report with visualizations (if requested)
3. Console output with summary statistics