"""Command-line interface for MSB"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List
import logging

from msb import MSBEvaluator, Config
from msb.datasets import list_available_datasets
from msb.models import list_available_models
from msb.utils import setup_logger, log_evaluation_start, log_evaluation_complete

logger = setup_logger(__name__)


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Multilingual Safety Benchmark (MSB) - Evaluate LLM safety across languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default configuration
  msb evaluate --config configs/default.yaml
  
  # Evaluate specific model on specific dataset
  msb evaluate --model gpt-4 --dataset multilingual_safety --languages en,zh,es
  
  # Compare multiple models
  msb compare --models gpt-4,claude-3-opus --dataset multilingual_safety
  
  # Generate report from existing results
  msb report --results results/evaluation_20240101.json
  
  # List available datasets and models
  msb list datasets
  msb list models
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version',
        version='MSB 1.0.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Run evaluation on a model'
    )
    eval_parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    eval_parser.add_argument(
        '--model',
        help='Model to evaluate (e.g., gpt-4, claude-3-opus)'
    )
    eval_parser.add_argument(
        '--dataset',
        default='multilingual_safety',
        help='Dataset to use (default: multilingual_safety)'
    )
    eval_parser.add_argument(
        '--languages',
        help='Comma-separated list of languages (e.g., en,zh,es)'
    )
    eval_parser.add_argument(
        '--metrics',
        help='Comma-separated list of metrics (e.g., safety,factuality,cultural)'
    )
    eval_parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum samples per language'
    )
    eval_parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results (default: results)'
    )
    eval_parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )
    eval_parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate HTML report after evaluation'
    )
    eval_parser.set_defaults(func=evaluate_command)
    
    # Compare command
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare multiple models'
    )
    compare_parser.add_argument(
        '--models',
        required=True,
        help='Comma-separated list of models to compare'
    )
    compare_parser.add_argument(
        '--dataset',
        default='multilingual_safety',
        help='Dataset to use'
    )
    compare_parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    compare_parser.add_argument(
        '--languages',
        help='Comma-separated list of languages'
    )
    compare_parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results'
    )
    compare_parser.set_defaults(func=compare_command)
    
    # Report command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate report from results'
    )
    report_parser.add_argument(
        '--results',
        required=True,
        help='Path to results JSON file'
    )
    report_parser.add_argument(
        '--format',
        choices=['html', 'markdown', 'pdf'],
        default='html',
        help='Report format (default: html)'
    )
    report_parser.add_argument(
        '--output-dir',
        default='reports',
        help='Output directory for report'
    )
    report_parser.set_defaults(func=report_command)
    
    # List command
    list_parser = subparsers.add_parser(
        'list',
        help='List available resources'
    )
    list_parser.add_argument(
        'resource',
        choices=['datasets', 'models', 'metrics'],
        help='Resource type to list'
    )
    list_parser.set_defaults(func=list_command)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate configuration or dataset'
    )
    validate_parser.add_argument(
        '--config',
        help='Configuration file to validate'
    )
    validate_parser.add_argument(
        '--dataset',
        help='Dataset file to validate'
    )
    validate_parser.set_defaults(func=validate_command)
    
    return parser


def evaluate_command(args):
    """Execute evaluation command"""
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.model:
        # Ensure model is configured
        if args.model not in config.models:
            config.models[args.model] = {
                "model_name": args.model
            }
    
    if args.languages:
        config.evaluation.languages = args.languages.split(',')
    
    if args.metrics:
        config.evaluation.metrics = args.metrics.split(',')
    
    if args.max_samples:
        config.evaluation.max_samples = args.max_samples
    
    if args.output_dir:
        config.evaluation.output_dir = args.output_dir
    
    # Validate configuration
    if not config.validate():
        logger.error("Invalid configuration")
        sys.exit(1)
    
    # Check if model is specified
    if not args.model:
        logger.error("Model must be specified (use --model)")
        sys.exit(1)
    
    # Create evaluator
    evaluator = MSBEvaluator(config)
    
    # Log start
    log_evaluation_start(
        args.model,
        args.dataset,
        config.evaluation.languages
    )
    
    try:
        # Run evaluation
        results = evaluator.evaluate(
            model=args.model,
            dataset=args.dataset,
            save_results=not args.no_save
        )
        
        # Log completion
        duration = 0  # Would calculate from timestamps
        total_samples = results.get("metadata", {}).get("total_samples", 0)
        log_evaluation_complete(duration, total_samples)
        
        # Generate report if requested
        if args.generate_report:
            report_path = evaluator.generate_report(results)
            logger.info(f"Report generated: {report_path}")
        
        # Print summary
        print_evaluation_summary(results)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def compare_command(args):
    """Execute model comparison command"""
    # Load configuration
    config = load_config(args.config)
    
    # Parse models
    models = args.models.split(',')
    
    # Override config if needed
    if args.languages:
        config.evaluation.languages = args.languages.split(',')
    
    if args.output_dir:
        config.evaluation.output_dir = args.output_dir
    
    # Create evaluator
    evaluator = MSBEvaluator(config)
    
    logger.info(f"Comparing models: {', '.join(models)}")
    
    try:
        # Run comparison
        comparison_results = evaluator.compare_models(
            models=models,
            dataset=args.dataset
        )
        
        # Save results
        output_path = Path(args.output_dir) / "comparison_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison results saved to: {output_path}")
        
        # Print summary
        print_comparison_summary(comparison_results)
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)


def report_command(args):
    """Execute report generation command"""
    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"Results file not found: {args.results}")
        sys.exit(1)
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Create temporary config for reporter
    from msb.utils import ReportGenerator
    reporter = ReportGenerator(None)
    
    try:
        # Generate report
        report_path = reporter.generate(
            results,
            args.output_dir,
            args.format
        )
        
        logger.info(f"Report generated: {report_path}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        sys.exit(1)


def list_command(args):
    """Execute list command"""
    if args.resource == 'datasets':
        datasets = list_available_datasets()
        print("\nAvailable Datasets:")
        print("-" * 50)
        for name, info in datasets.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Languages: {', '.join(info['languages'])}")
            print(f"  Categories: {', '.join(info['categories'])}")
            print(f"  Size: {info['size']}")
    
    elif args.resource == 'models':
        models = list_available_models()
        print("\nAvailable Models:")
        print("-" * 50)
        for provider, model_list in models.items():
            print(f"\n{provider.upper()}:")
            for model in model_list:
                print(f"  - {model}")
    
    elif args.resource == 'metrics':
        print("\nAvailable Metrics:")
        print("-" * 50)
        print("\nsafety:")
        print("  - Toxicity detection")
        print("  - Harm probability assessment")
        print("  - Bias detection")
        print("\nfactuality:")
        print("  - Accuracy verification")
        print("  - Hallucination detection")
        print("  - Source reliability")
        print("\ncultural:")
        print("  - Cultural appropriateness")
        print("  - Stereotype avoidance")
        print("  - Regional sensitivity")


def validate_command(args):
    """Execute validation command"""
    if args.config:
        try:
            config = Config(args.config)
            if config.validate():
                logger.info("Configuration is valid")
            else:
                logger.error("Configuration validation failed")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    if args.dataset:
        try:
            from msb.datasets import get_dataset, validate_dataset
            dataset = get_dataset(args.dataset)
            if validate_dataset(dataset):
                logger.info("Dataset is valid")
                stats = dataset.get_statistics()
                print(f"Total samples: {stats['total_samples']}")
                print(f"Languages: {', '.join(stats['languages'].keys())}")
            else:
                logger.error("Dataset validation failed")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)


def load_config(config_path: Optional[str]) -> Config:
    """Load configuration from file or create default"""
    if config_path:
        return Config(config_path)
    else:
        # Create default configuration
        config = Config()
        # Set some defaults
        config.evaluation.languages = ["en"]
        config.evaluation.metrics = ["safety", "factuality", "cultural"]
        return config


def print_evaluation_summary(results: dict):
    """Print evaluation summary to console"""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"Model: {results.get('model', 'Unknown')}")
    print(f"Dataset: {results.get('dataset', 'Unknown')}")
    
    # Print aggregate metrics
    if "aggregate" in results and "metrics" in results["aggregate"]:
        print("\nOverall Metrics:")
        for metric, stats in results["aggregate"]["metrics"].items():
            if isinstance(stats, dict) and "mean" in stats:
                print(f"  {metric}: {stats['mean']:.3f} (±{stats['std']:.3f})")
    
    # Print language breakdown
    if "languages" in results:
        print("\nLanguage Performance:")
        for lang, data in results["languages"].items():
            if isinstance(data, dict) and "num_samples" in data:
                print(f"  {lang}: {data['num_samples']} samples")


def print_comparison_summary(results: dict):
    """Print model comparison summary"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    if "comparison" in results and "rankings" in results["comparison"]:
        rankings = results["comparison"]["rankings"]
        for metric, ranking in rankings.items():
            print(f"\n{metric.upper()} Rankings:")
            for i, model in enumerate(ranking, 1):
                print(f"  {i}. {model}")


if __name__ == "__main__":
    main()