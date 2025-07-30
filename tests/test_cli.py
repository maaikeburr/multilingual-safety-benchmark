"""Tests for CLI functionality"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

from msb.cli import (
    main, 
    evaluate_command,
    compare_command,
    report_command,
    list_command,
    validate_command,
    create_parser
)


class TestCLI:
    """Test command-line interface"""
    
    def test_create_parser(self):
        """Test parser creation"""
        parser = create_parser()
        
        # Test parser has subcommands
        args = parser.parse_args(['evaluate', '--model', 'gpt-4'])
        assert args.command == 'evaluate'
        assert args.model == 'gpt-4'
    
    def test_parser_evaluate_command(self):
        """Test evaluate command parsing"""
        parser = create_parser()
        
        args = parser.parse_args([
            'evaluate',
            '--model', 'gpt-4',
            '--dataset', 'test_dataset',
            '--languages', 'en,zh',
            '--max-samples', '10',
            '--no-save'
        ])
        
        assert args.model == 'gpt-4'
        assert args.dataset == 'test_dataset'
        assert args.languages == 'en,zh'
        assert args.max_samples == 10
        assert args.no_save == True
    
    @patch('msb.cli.MSBEvaluator')
    @patch('msb.cli.load_config')
    def test_evaluate_command_execution(self, mock_load_config, mock_evaluator_class):
        """Test evaluate command execution"""
        # Setup mocks
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config.models = {}
        mock_config.evaluation = Mock(languages=['en'], metrics=['safety'])
        mock_load_config.return_value = mock_config
        
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {
            "model": "gpt-4",
            "metadata": {"total_samples": 10},
            "aggregate": {"metrics": {"safety": {"mean": 0.85}}}
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        # Create args
        args = Mock()
        args.model = "gpt-4"
        args.dataset = "test_dataset"
        args.config = None
        args.languages = "en,zh"
        args.metrics = None
        args.max_samples = None
        args.output_dir = None
        args.no_save = False
        args.generate_report = False
        
        # Execute command
        evaluate_command(args)
        
        # Verify
        mock_evaluator.evaluate.assert_called_once()
        assert mock_config.evaluation.languages == ["en", "zh"]
    
    @patch('msb.cli.json.dump')
    @patch('msb.cli.open', create=True)
    @patch('msb.cli.MSBEvaluator')
    def test_compare_command(self, mock_evaluator_class, mock_open, mock_json_dump):
        """Test compare command"""
        # Setup mocks
        mock_evaluator = Mock()
        mock_evaluator.compare_models.return_value = {
            "models": ["gpt-4", "claude-3"],
            "comparison": {"rankings": {"safety": ["gpt-4", "claude-3"]}}
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        args = Mock()
        args.models = "gpt-4,claude-3"
        args.dataset = "test_dataset"
        args.config = None
        args.languages = None
        args.output_dir = "results"
        
        # Execute
        compare_command(args)
        
        # Verify
        mock_evaluator.compare_models.assert_called_with(
            models=["gpt-4", "claude-3"],
            dataset="test_dataset"
        )
    
    @patch('msb.cli.ReportGenerator')
    @patch('msb.cli.json.load')
    @patch('msb.cli.open', create=True)
    @patch('msb.cli.Path')
    def test_report_command(self, mock_path_class, mock_open, mock_json_load, mock_reporter_class):
        """Test report command"""
        # Setup mocks
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path_class.return_value = mock_path
        
        mock_json_load.return_value = {"model": "gpt-4", "results": {}}
        
        mock_reporter = Mock()
        mock_reporter.generate.return_value = "report.html"
        mock_reporter_class.return_value = mock_reporter
        
        args = Mock()
        args.results = "results.json"
        args.format = "html"
        args.output_dir = "reports"
        
        # Execute
        report_command(args)
        
        # Verify
        mock_reporter.generate.assert_called_once()
    
    @patch('msb.cli.list_available_models')
    @patch('msb.cli.list_available_datasets')
    def test_list_command(self, mock_list_datasets, mock_list_models):
        """Test list command"""
        # Mock return values
        mock_list_datasets.return_value = {
            "test_dataset": {
                "description": "Test",
                "languages": ["en"],
                "categories": ["test"],
                "size": "small"
            }
        }
        mock_list_models.return_value = {
            "openai": ["gpt-4", "gpt-3.5-turbo"]
        }
        
        # Test listing datasets
        args = Mock(resource="datasets")
        list_command(args)
        mock_list_datasets.assert_called_once()
        
        # Test listing models
        args = Mock(resource="models")
        list_command(args)
        mock_list_models.assert_called_once()
    
    @patch('msb.cli.Config')
    def test_validate_command_config(self, mock_config_class):
        """Test validate command for config"""
        # Setup mock
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config_class.return_value = mock_config
        
        args = Mock()
        args.config = "config.yaml"
        args.dataset = None
        
        # Execute - should not raise
        validate_command(args)
        
        # Test invalid config
        mock_config.validate.return_value = False
        with pytest.raises(SystemExit):
            validate_command(args)