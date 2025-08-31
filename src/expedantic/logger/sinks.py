"""Sink implementations for the logger system.

This module contains all the built-in sink implementations that can
receive and process logged data from loggers.
"""

import json
import threading
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Protocol, Any

from .fields import SupportedTypes


class SinkProtocol(Protocol):
    """Protocol defining the interface for logger sinks.
    
    Sinks receive flushed data from loggers and output it to various destinations
    like console, files, databases, or remote logging services.
    """
    
    def write(self, data: dict[str, SupportedTypes], logger_name: str | None = None) -> None:
        """Write a single log entry to this sink.
        
        Args:
            data: The log entry data containing field values and timestamp
            logger_name: Optional name of the logger that produced this entry
        """
        ...
    
    def close(self) -> None:
        """Close the sink and release any resources.
        
        Called when the logger is being destroyed or explicitly closed.
        """
        ...


class ConsoleSink:
    """Sink that outputs log entries to the console using Rich formatting.
    
    Provides clean, readable output suitable for development and debugging.
    """
    
    def __init__(self, 
                 show_timestamp: bool = True,
                 show_logger_name: bool = True,
                 timestamp_format: str = "%H:%M:%S"):
        """Initialize the console sink.
        
        Args:
            show_timestamp: Whether to display timestamps in output
            show_logger_name: Whether to display logger names in output  
            timestamp_format: strftime format for timestamp display
        """
        self.show_timestamp = show_timestamp
        self.show_logger_name = show_logger_name
        self.timestamp_format = timestamp_format
        
    def write(self, data: dict[str, SupportedTypes], logger_name: str | None = None) -> None:
        """Write log entry to console with Rich formatting."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        
        console = Console()
        
        # Build display components
        parts = []
        
        if self.show_timestamp and '_timestamp' in data:
            timestamp = data['_timestamp']
            if isinstance(timestamp, datetime):
                time_str = timestamp.strftime(self.timestamp_format)
                parts.append(f"[dim]{time_str}[/dim]")
        
        if self.show_logger_name and logger_name:
            parts.append(f"[bold blue]{logger_name}[/bold blue]")
            
        # Format the data (excluding timestamp since we handle it separately)
        data_items = {k: v for k, v in data.items() if k != '_timestamp'}
        
        if data_items:
            formatted_data = []
            for key, value in data_items.items():
                if isinstance(value, float):
                    formatted_data.append(f"{key}={value:.4f}")
                else:
                    formatted_data.append(f"{key}={value}")
            parts.append(" ".join(formatted_data))
        
        # Output the log line
        if parts:
            console.print(" | ".join(parts))
    
    def close(self) -> None:
        """Console sink has no resources to close."""
        pass


class FileSink:
    """Sink that appends log entries to a file in JSON Lines format.
    
    Each log entry is written as a single JSON object per line,
    making it easy to parse and process with standard tools.
    """
    
    def __init__(self, 
                 filepath: str | Path,
                 mode: str = "a",
                 encoding: str = "utf-8"):
        """Initialize the file sink.
        
        Args:
            filepath: Path to the output file
            mode: File open mode (usually 'a' for append or 'w' for overwrite)
            encoding: Text encoding for the file
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self.encoding = encoding
        self._file = None
        self._lock = threading.Lock()
        
    def _ensure_open(self):
        """Ensure the file is open for writing."""
        if self._file is None or self._file.closed:
            self._file = open(self.filepath, self.mode, encoding=self.encoding)
    
    def write(self, data: dict[str, SupportedTypes], logger_name: str | None = None) -> None:
        """Write log entry to file as JSON."""
        with self._lock:
            self._ensure_open()
            
            # Prepare data for JSON serialization
            json_data = {}
            if logger_name:
                json_data['_logger'] = logger_name
                
            for key, value in data.items():
                if isinstance(value, datetime):
                    json_data[key] = value.isoformat()
                elif isinstance(value, (date, time, timedelta)):
                    json_data[key] = str(value)
                elif isinstance(value, Decimal):
                    json_data[key] = float(value)
                elif isinstance(value, (list, tuple)):
                    # Convert complex nested types to strings if needed
                    json_data[key] = [str(item) if not isinstance(item, (int, float, bool, str, type(None))) else item 
                                     for item in value]
                else:
                    json_data[key] = value
            
            # Write as JSON line
            json.dump(json_data, self._file, separators=(',', ':'))
            self._file.write('\n')
            self._file.flush()
    
    def close(self) -> None:
        """Close the file handle."""
        with self._lock:
            if self._file and not self._file.closed:
                self._file.close()


class WandBSink:
    """Sink that logs entries to Weights & Biases.
    
    Automatically handles different data types and creates appropriate
    WandB log entries. Requires wandb to be installed and initialized.
    """
    
    def __init__(self, 
                 project: str | None = None,
                 step_field: str = "iteration",
                 prefix: str = ""):
        """Initialize the WandB sink.
        
        Args:
            project: WandB project name (if not already initialized)
            step_field: Field name to use as the step counter for WandB
            prefix: Optional prefix to add to all metric names
        """
        self.step_field = step_field
        self.prefix = prefix
        self._wandb = None
        
        # Lazy import and initialization
        try:
            import wandb
            self._wandb = wandb
            if project and not wandb.run:
                wandb.init(project=project)
        except ImportError:
            raise ImportError("wandb package is required for WandBSink. Install with: pip install wandb")
    
    def write(self, data: dict[str, SupportedTypes], logger_name: str | None = None) -> None:
        """Write log entry to WandB."""
        if not self._wandb or not self._wandb.run:
            return  # Skip if WandB is not initialized
            
        # Prepare WandB log data
        wandb_data = {}
        step = None
        
        for key, value in data.items():
            if key == '_timestamp':
                continue  # WandB handles timestamps automatically
                
            # Extract step if present
            if key == self.step_field:
                step = value
                
            # Add prefix if specified
            metric_name = f"{self.prefix}{key}" if self.prefix else key
            
            # Convert values for WandB compatibility
            if isinstance(value, (datetime, date, time, timedelta, Decimal)):
                wandb_data[metric_name] = str(value)
            elif isinstance(value, (list, tuple)):
                # WandB can handle lists of numbers
                if value and all(isinstance(x, (int, float)) for x in value):
                    wandb_data[metric_name] = value
                else:
                    wandb_data[metric_name] = str(value)
            else:
                wandb_data[metric_name] = value
        
        # Log to WandB
        if wandb_data:
            if step is not None:
                self._wandb.log(wandb_data, step=step)
            else:
                self._wandb.log(wandb_data)
    
    def close(self) -> None:
        """WandB handles cleanup automatically."""
        pass


class TensorBoardSink:
    """Sink that logs scalar metrics to TensorBoard.
    
    Creates TensorBoard-compatible log files that can be viewed with
    the TensorBoard web interface. Requires tensorboardX or tensorflow.
    """
    
    def __init__(self, 
                 log_dir: str | Path,
                 step_field: str = "iteration"):
        """Initialize the TensorBoard sink.
        
        Args:
            log_dir: Directory to store TensorBoard log files
            step_field: Field name to use as the step counter
        """
        self.log_dir = Path(log_dir)
        self.step_field = step_field
        self._writer = None
        
        # Try to import TensorBoard writer
        try:
            from tensorboardX import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                raise ImportError("tensorboardX or torch is required for TensorBoardSink. "
                                "Install with: pip install tensorboardX")
    
    def write(self, data: dict[str, SupportedTypes], logger_name: str | None = None) -> None:
        """Write scalar metrics to TensorBoard."""
        if not self._writer:
            return
            
        step = None
        
        # Extract step value
        if self.step_field in data:
            step = data[self.step_field]
            if not isinstance(step, int):
                step = int(step) if step is not None else None
        
        # Log scalar metrics
        for key, value in data.items():
            if key in ('_timestamp', self.step_field):
                continue
                
            # Only log numeric scalars to TensorBoard
            if isinstance(value, (int, float, bool)):
                tag = f"{logger_name}/{key}" if logger_name else key
                self._writer.add_scalar(tag, float(value), global_step=step)
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self._writer:
            self._writer.close()