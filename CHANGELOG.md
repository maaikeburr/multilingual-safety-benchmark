# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-01

### Added
- Initial release of Multilingual Safety Benchmark (MSB)
- Core evaluation engine with batch processing and parallelization
- Support for OpenAI, Anthropic, and Cohere models
- Three evaluation metrics: Safety, Factuality, and Cultural Sensitivity
- Built-in multilingual safety dataset with 5 languages
- Support for custom datasets (JSON, CSV, JSONL)
- HuggingFace datasets integration
- HTML and Markdown report generation with visualizations
- Command-line interface with multiple commands
- Comprehensive configuration system with YAML support
- Statistical analysis and model comparison features
- Rich logging with file and console output
- Retry logic and error handling for API calls
- Example scripts and quickstart guide
- Full API documentation
- Unit test framework

### Security
- API keys managed through environment variables
- No sensitive data logged or saved in results
- Safe file path handling

## [Unreleased]

### Planned
- PDF report generation
- Additional language support (10+ languages)
- Real-time evaluation monitoring
- Web-based dashboard
- Plugin system for custom models
- Advanced caching mechanisms
- Distributed evaluation support
- Integration with more model providers
- Enhanced visualization options
- Automated benchmark leaderboard