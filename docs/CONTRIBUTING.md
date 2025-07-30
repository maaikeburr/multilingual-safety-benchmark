# Contributing to MSB

We welcome contributions to the Multilingual Safety Benchmark! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/msb-complete.git
   cd msb-complete
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Process

### 1. Create a Branch

Create a feature branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Run checks:
```bash
black msb tests
flake8 msb tests
mypy msb
```

### 4. Testing

Run tests with pytest:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=msb

# Run specific test file
pytest tests/test_evaluator.py
```

### 5. Documentation

- Update docstrings for new functions/classes
- Update API reference if adding public APIs
- Add examples for new features

### 6. Commit Messages

Follow conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `style:` Code style changes
- `chore:` Maintenance tasks

Example:
```
feat: add support for HuggingFace datasets

- Implement HuggingFaceDataset class
- Add automatic format conversion
- Include tests and documentation
```

## Adding New Features

### Adding a New Model

1. Create a new file in `msb/models/`
2. Inherit from `BaseModel`
3. Implement required methods:
   ```python
   from msb.models.base import BaseModel
   
   class NewModel(BaseModel):
       def _initialize(self):
           # Setup model-specific configuration
           pass
       
       def generate(self, prompt, **kwargs):
           # Generate response
           return response
   ```
4. Register in `msb/models/__init__.py`
5. Add tests in `tests/models/`
6. Update documentation

### Adding a New Metric

1. Create evaluator in `msb/evaluation/`
2. Implement evaluation logic:
   ```python
   @dataclass
   class NewMetricScore:
       score: float
       details: Dict[str, Any]
       
       def to_dict(self):
           return {"score": self.score, "details": self.details}
   
   class NewMetricEvaluator:
       def evaluate(self, prompt, response, context, language):
           # Compute metric
           return NewMetricScore(score=0.9, details={})
   ```
3. Add to evaluator registry
4. Add tests and documentation

### Adding a Dataset Format

1. Create loader in `msb/datasets/`
2. Inherit from `BaseDataset`
3. Implement data loading:
   ```python
   class NewFormatDataset(BaseDataset):
       def load(self):
           # Load data into self.data
           pass
       
       def get_samples(self, language, max_samples=None):
           # Return samples for language
           pass
   ```
4. Register in dataset loaders
5. Add tests and examples

## Testing Guidelines

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── test_evaluator.py    # Core tests
├── models/
│   ├── test_openai.py
│   └── test_anthropic.py
├── datasets/
│   └── test_loaders.py
├── evaluation/
│   ├── test_safety.py
│   └── test_factuality.py
└── utils/
    └── test_metrics.py
```

### Writing Tests

- Use pytest fixtures for common setup
- Mock external API calls
- Test edge cases and error handling
- Aim for >80% code coverage

Example test:
```python
def test_safety_evaluator():
    config = {"toxicity_threshold": 0.7}
    evaluator = SafetyEvaluator(config)
    
    score = evaluator.evaluate(
        prompt="Test prompt",
        response="Safe response",
        language="en"
    )
    
    assert score.overall_safety > 0.8
    assert score.toxicity < 0.3
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG.md
4. Submit pull request with clear description
5. Address reviewer feedback

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog entry added
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main

## Release Process

1. Update version in `setup.py` and `msb/__init__.py`
2. Update CHANGELOG.md
3. Create release PR
4. After merge, tag release:
   ```bash
   git tag -a v1.0.1 -m "Release version 1.0.1"
   git push origin v1.0.1
   ```

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions in GitHub Discussions
- Check existing issues before creating new ones

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Follow the project's code of conduct

Thank you for contributing to MSB!