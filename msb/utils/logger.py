"""Logging utilities for MSB"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install as install_rich_traceback

# Install rich traceback for better error messages
install_rich_traceback()

# Create console for rich output
console = Console()


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True
) -> logging.Logger:
    """
    Setup a logger with optional file output and rich formatting
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        use_rich: Whether to use rich formatting for console output
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    if use_rich:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
    
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    return logging.getLogger(name)


def log_evaluation_start(model: str, dataset: str, languages: list) -> None:
    """Log the start of an evaluation"""
    console.print(f"\n[bold blue]Starting Evaluation[/bold blue]")
    console.print(f"Model: [green]{model}[/green]")
    console.print(f"Dataset: [green]{dataset}[/green]")
    console.print(f"Languages: [green]{', '.join(languages)}[/green]")
    console.print(f"Start Time: [yellow]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/yellow]\n")


def log_evaluation_progress(current: int, total: int, language: str) -> None:
    """Log evaluation progress"""
    percentage = (current / total) * 100
    console.print(
        f"[cyan]Progress[/cyan]: {current}/{total} ({percentage:.1f}%) - "
        f"Language: [yellow]{language}[/yellow]",
        end="\r"
    )


def log_evaluation_complete(duration: float, total_samples: int) -> None:
    """Log evaluation completion"""
    console.print(f"\n[bold green]Evaluation Complete![/bold green]")
    console.print(f"Duration: [yellow]{duration:.2f} seconds[/yellow]")
    console.print(f"Total Samples: [yellow]{total_samples}[/yellow]")
    console.print(f"Samples/second: [yellow]{total_samples/duration:.2f}[/yellow]\n")


def log_error(error: Exception, context: str = "") -> None:
    """Log an error with context"""
    console.print(f"\n[bold red]Error Occurred[/bold red]")
    if context:
        console.print(f"Context: [yellow]{context}[/yellow]")
    console.print(f"Error Type: [red]{type(error).__name__}[/red]")
    console.print(f"Error Message: [red]{str(error)}[/red]\n")


def create_evaluation_logger(
    output_dir: str,
    run_id: Optional[str] = None
) -> logging.Logger:
    """
    Create a logger specifically for evaluation runs
    
    Args:
        output_dir: Directory for log files
        run_id: Optional run identifier
        
    Returns:
        Configured logger
    """
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(output_dir) / f"evaluation_{run_id}.log"
    
    return setup_logger(
        name=f"msb.evaluation.{run_id}",
        level="INFO",
        log_file=str(log_file),
        use_rich=True
    )