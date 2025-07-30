# API Reference

## Core Classes

### MSBEvaluator

Main evaluation class for running benchmarks.

```python
from msb import MSBEvaluator

evaluator = MSBEvaluator(config="path/to/config.yaml")
```

#### Methods

##### evaluate()
```python
def evaluate(
    model: str,
    dataset: str,
    languages: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    save_results: bool = True
) -> Dict[str, Any]
```

Run evaluation on a model.

**Parameters:**
- `model`: Model name or identifier
- `dataset`: Dataset name or path
- `languages`: List of language codes (default: from config)
- `metrics`: List of metrics to compute (default: from config)
- `max_samples`: Maximum samples per language
- `save_results`: Whether to save results to disk

**Returns:**
- Dictionary containing evaluation results

##### compare_models()
```python
def compare_models(
    models: List[str],
    dataset: str,
    **kwargs
) -> Dict[str, Any]
```

Compare multiple models on the same dataset.

**Parameters:**
- `models`: List of model names
- `dataset`: Dataset to use
- `**kwargs`: Additional arguments passed to evaluate()

**Returns:**
- Comparison results with rankings

##### generate_report()
```python
def generate_report(
    results: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    format: str = "html"
) -> str
```

Generate evaluation report.

**Parameters:**
- `results`: Results dictionary (uses self.results if None)
- `output_dir`: Output directory for report
- `format`: Report format ("html", "markdown", "pdf")

**Returns:**
- Path to generated report

### Config

Configuration management class.

```python
from msb.core import Config

config = Config("config.yaml")
```

#### Methods

##### load()
```python
def load(config_path: str) -> None
```

Load configuration from YAML file.

##### validate()
```python
def validate() -> bool
```

Validate configuration completeness and correctness.

##### get_model_config()
```python
def get_model_config(model_name: str) -> Optional[ModelConfig]
```

Get configuration for a specific model.

## Model Classes

### BaseModel

Abstract base class for all models.

```python
from msb.models import BaseModel

class MyModel(BaseModel):
    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation
        pass
```

#### Abstract Methods

##### generate()
```python
@abstractmethod
def generate(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> str
```

Generate response for a single prompt.

#### Concrete Methods

##### generate_batch()
```python
def generate_batch(
    prompts: List[str],
    max_workers: int = 5,
    **kwargs
) -> List[Optional[str]]
```

Generate responses for multiple prompts in parallel.

## Dataset Classes

### BaseDataset

Abstract base class for datasets.

```python
from msb.datasets import BaseDataset

class MyDataset(BaseDataset):
    def load(self) -> None:
        # Load dataset
        pass
    
    def get_samples(self, language: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        # Return samples
        pass
```

#### Abstract Methods

##### load()
```python
@abstractmethod
def load() -> None
```

Load the dataset.

##### get_samples()
```python
@abstractmethod
def get_samples(language: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]
```

Get samples for a specific language.

#### Concrete Methods

##### get_languages()
```python
def get_languages() -> List[str]
```

Get list of available languages.

##### get_statistics()
```python
def get_statistics() -> Dict[str, Any]
```

Get dataset statistics.

## Evaluation Classes

### SafetyEvaluator

Evaluator for safety metrics.

```python
from msb.evaluation import SafetyEvaluator

evaluator = SafetyEvaluator(config)
score = evaluator.evaluate(prompt, response, context, language)
```

#### Methods

##### evaluate()
```python
def evaluate(
    prompt: str,
    response: str,
    context: Optional[Dict[str, Any]] = None,
    language: str = "en"
) -> SafetyScore
```

Evaluate safety of a model response.

**Returns:**
- SafetyScore object with:
  - `toxicity`: Toxicity level (0-1)
  - `harm_probability`: Probability of harmful content (0-1)
  - `bias_score`: Bias detection score (0-1, higher is better)
  - `overall_safety`: Combined safety score (0-1)

### FactualityEvaluator

Evaluator for factuality metrics.

```python
from msb.evaluation import FactualityEvaluator

evaluator = FactualityEvaluator(config)
score = evaluator.evaluate(prompt, response, context, language)
```

#### Methods

##### evaluate()
```python
def evaluate(
    prompt: str,
    response: str,
    context: Optional[Dict[str, Any]] = None,
    language: str = "en"
) -> FactualityScore
```

Evaluate factuality of a model response.

**Returns:**
- FactualityScore object with:
  - `accuracy`: Factual accuracy (0-1)
  - `hallucination_rate`: Rate of hallucinations (0-1)
  - `source_reliability`: Source citation quality (0-1)
  - `overall_factuality`: Combined factuality score (0-1)

### CulturalEvaluator

Evaluator for cultural sensitivity metrics.

```python
from msb.evaluation import CulturalEvaluator

evaluator = CulturalEvaluator(config)
score = evaluator.evaluate(prompt, response, context, language)
```

#### Methods

##### evaluate()
```python
def evaluate(
    prompt: str,
    response: str,
    context: Optional[Dict[str, Any]] = None,
    language: str = "en"
) -> CulturalScore
```

Evaluate cultural sensitivity of a model response.

**Returns:**
- CulturalScore object with:
  - `appropriateness`: Cultural appropriateness (0-1)
  - `stereotype_avoidance`: Stereotype avoidance score (0-1)
  - `cultural_awareness`: Cultural awareness score (0-1)
  - `overall_cultural_score`: Combined cultural score (0-1)

## Utility Functions

### Metrics

```python
from msb.utils import calculate_statistics, compare_distributions

# Calculate statistics
stats = calculate_statistics(scores)

# Compare two score distributions
comparison = compare_distributions(scores1, scores2, test="mann-whitney")
```

### Logging

```python
from msb.utils import setup_logger, log_evaluation_start

# Setup logger
logger = setup_logger("my_module", level="INFO", log_file="output.log")

# Log evaluation progress
log_evaluation_start(model="gpt-4", dataset="safety", languages=["en", "zh"])
```

## Command Line Interface

### evaluate

Run evaluation on a model.

```bash
msb evaluate --model gpt-4 --dataset multilingual_safety --languages en,zh,es
```

Options:
- `--config`: Configuration file path
- `--model`: Model to evaluate
- `--dataset`: Dataset to use
- `--languages`: Comma-separated language codes
- `--metrics`: Comma-separated metrics
- `--max-samples`: Maximum samples per language
- `--output-dir`: Output directory
- `--no-save`: Don't save results
- `--generate-report`: Generate HTML report

### compare

Compare multiple models.

```bash
msb compare --models gpt-4,claude-3-opus --dataset multilingual_safety
```

Options:
- `--models`: Comma-separated model names
- `--dataset`: Dataset to use
- `--config`: Configuration file
- `--languages`: Language codes
- `--output-dir`: Output directory

### report

Generate report from results.

```bash
msb report --results results/evaluation.json --format html
```

Options:
- `--results`: Path to results JSON
- `--format`: Output format (html, markdown, pdf)
- `--output-dir`: Output directory

### list

List available resources.

```bash
msb list datasets
msb list models
msb list metrics
```

### validate

Validate configuration or dataset.

```bash
msb validate --config config.yaml
msb validate --dataset data/custom.json
```