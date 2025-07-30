# MSB Architecture Documentation

## Overview

The Multilingual Safety Benchmark (MSB) is designed with a modular architecture that separates concerns and allows for easy extension. This document describes the key components and their interactions.

## Core Components

### 1. Configuration System (`msb.core.config`)

The configuration system manages all settings for the framework:

- **Config**: Main configuration class that loads from YAML files
- **ModelConfig**: Model-specific settings (API keys, parameters)
- **EvaluationConfig**: Evaluation settings (batch size, languages, metrics)
- **MetricConfig**: Metric-specific thresholds and parameters

Configuration supports environment variable substitution for sensitive data like API keys.

### 2. Evaluation Engine (`msb.core.evaluator`)

The MSBEvaluator is the main orchestrator that:

- Loads models and datasets
- Manages the evaluation workflow
- Handles batch processing and parallelization
- Aggregates results across languages and metrics
- Generates reports

Key methods:
- `evaluate()`: Run evaluation on a single model
- `compare_models()`: Compare multiple models
- `generate_report()`: Create evaluation reports

### 3. Model Interfaces (`msb.models`)

Abstract base class `BaseModel` defines the interface for all model implementations:

- `generate()`: Generate response for single prompt
- `generate_batch()`: Batch generation with parallelization
- Built-in retry logic and error handling

Implementations:
- **OpenAIModel**: GPT-3.5, GPT-4, etc.
- **AnthropicModel**: Claude family
- **CohereModel**: Command models

### 4. Dataset System (`msb.datasets`)

Flexible dataset loading supporting multiple formats:

- **BaseDataset**: Abstract interface for all datasets
- **MultilingualSafetyDataset**: Built-in safety evaluation dataset
- **HuggingFaceDataset**: Integration with HuggingFace datasets
- **CustomDataset**: Load from JSON, CSV, or JSONL files

### 5. Evaluation Metrics (`msb.evaluation`)

Three core metric categories:

#### Safety Metrics
- Toxicity detection
- Harm category classification
- Bias detection
- Refusal appropriateness

#### Factuality Metrics
- Claim verification
- Hallucination detection
- Source reliability
- Knowledge accuracy

#### Cultural Metrics
- Cultural appropriateness
- Stereotype detection
- Regional sensitivity
- Language-specific norms

### 6. Utilities (`msb.utils`)

Supporting utilities:

- **Logger**: Rich logging with file and console output
- **Metrics**: Statistical calculations and comparisons
- **Reporter**: HTML/Markdown/PDF report generation

## Data Flow

```
1. User Configuration
   ↓
2. MSBEvaluator Initialization
   ↓
3. Load Model & Dataset
   ↓
4. For each language:
   a. Get samples from dataset
   b. Generate model responses (batched)
   c. Evaluate with each metric
   d. Aggregate scores
   ↓
5. Compute overall statistics
   ↓
6. Generate report
```

## Extension Points

### Adding a New Model

1. Create a class inheriting from `BaseModel`
2. Implement the `generate()` method
3. Register in `models/__init__.py`

Example:
```python
class MyModel(BaseModel):
    def generate(self, prompt, **kwargs):
        # Your implementation
        return response
```

### Adding a New Metric

1. Create evaluator class with `evaluate()` method
2. Return a score object with `to_dict()` method
3. Register in evaluation system

Example:
```python
class MyMetricEvaluator:
    def evaluate(self, prompt, response, context, language):
        # Calculate scores
        return MyScore(...)
```

### Adding a New Dataset Format

1. Inherit from `BaseDataset`
2. Implement `load()` and `get_samples()`
3. Register in dataset loaders

## Performance Considerations

- **Batch Processing**: Use appropriate batch sizes for API rate limits
- **Parallelization**: Concurrent API calls with configurable workers
- **Caching**: Results are saved incrementally
- **Memory**: Large datasets are processed in chunks

## Error Handling

- All API calls have retry logic with exponential backoff
- Errors are logged but don't stop evaluation
- Partial results are saved in case of interruption
- Comprehensive error reporting in final results

## Security Considerations

- API keys are never logged or saved in results
- Sensitive prompts/responses can be filtered
- All file I/O uses safe path handling
- Input validation on all user data