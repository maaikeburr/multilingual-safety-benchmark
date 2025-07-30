# Multilingual Safety Benchmark (MSB) - Complete Edition

A comprehensive framework for evaluating Large Language Model (LLM) safety, factuality, and alignment across multilingual and culturally sensitive contexts.

## 🌟 Features

- **Multilingual Support**: Evaluate models across 50+ languages
- **Cultural Sensitivity**: Assess cultural appropriateness and bias
- **Safety Metrics**: Comprehensive safety evaluation including toxicity, harmfulness, and misinformation
- **Model Agnostic**: Support for OpenAI, Anthropic, Cohere, and custom models
- **Flexible Configuration**: YAML-based configuration system
- **Extensible Architecture**: Easy to add new metrics, datasets, and models
- **Detailed Reporting**: Generate comprehensive evaluation reports with visualizations

## 📋 Requirements

- Python 3.8+
- API keys for the models you want to evaluate (OpenAI, Anthropic, etc.)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/msb-complete.git
cd msb-complete

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from msb import MSBEvaluator

# Initialize evaluator
evaluator = MSBEvaluator(config_path="configs/default.yaml")

# Run evaluation
results = evaluator.evaluate(
    model="claude-3-opus-20240229",
    dataset="multilingual_safety",
    languages=["en", "zh", "es", "ar", "hi"]
)

# Generate report
evaluator.generate_report(results, output_dir="results/")
```

### Command Line Interface

```bash
# Run evaluation with default configuration
python -m msb evaluate --config configs/default.yaml

# Evaluate specific model on specific dataset
python -m msb evaluate --model gpt-4 --dataset multilingual_safety --languages en,zh,es

# Generate report from existing results
python -m msb report --results results/evaluation_20240101.json
```

## 📁 Project Structure

```
msb-complete/
├── msb/                    # Core package
│   ├── core/              # Core functionality
│   │   ├── __init__.py
│   │   ├── evaluator.py   # Main evaluation engine
│   │   └── config.py      # Configuration management
│   ├── evaluation/        # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── safety.py      # Safety metrics
│   │   ├── factuality.py  # Factuality checks
│   │   └── cultural.py    # Cultural sensitivity
│   ├── models/            # Model interfaces
│   │   ├── __init__.py
│   │   ├── base.py        # Base model interface
│   │   ├── openai.py      # OpenAI integration
│   │   ├── anthropic.py   # Anthropic integration
│   │   └── cohere.py      # Cohere integration
│   ├── datasets/          # Dataset loaders
│   │   ├── __init__.py
│   │   ├── base.py        # Base dataset class
│   │   └── loaders.py     # Dataset loading utilities
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── logger.py      # Logging utilities
│       └── metrics.py     # Metric calculations
├── configs/               # Configuration files
├── data/                  # Data directory
├── docs/                  # Documentation
├── examples/              # Example scripts
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
└── requirements.txt       # Dependencies
```

## 🔧 Configuration

The framework uses YAML configuration files. See `configs/default.yaml` for an example:

```yaml
evaluation:
  batch_size: 32
  max_retries: 3
  timeout: 30

models:
  openai:
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-3-opus-20240229

metrics:
  safety:
    toxicity_threshold: 0.7
    harm_categories: ["violence", "self-harm", "sexual", "hate"]
  factuality:
    fact_checking_model: "gpt-4"
    confidence_threshold: 0.8
```

## 📊 Evaluation Metrics

### Safety Metrics
- **Toxicity Score**: Measures harmful or offensive content
- **Harm Probability**: Likelihood of generating harmful outputs
- **Bias Detection**: Identifies demographic and cultural biases

### Factuality Metrics
- **Accuracy Score**: Correctness of factual claims
- **Hallucination Rate**: Frequency of fabricated information
- **Source Verification**: Ability to cite reliable sources

### Cultural Metrics
- **Cultural Appropriateness**: Respect for cultural norms
- **Linguistic Accuracy**: Proper use of language nuances
- **Context Sensitivity**: Understanding of cultural context

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research is supported by the AI Safety community and various academic institutions. Special thanks to all contributors and researchers in the field of multilingual AI evaluation.

## 📚 Citation

If you use MSB in your research, please cite:

```bibtex
@software{msb2024,
  title = {Multilingual Safety Benchmark},
  author = {MSB Contributors},
  year = {2024},
  url = {https://github.com/your-org/msb-complete}
}
```