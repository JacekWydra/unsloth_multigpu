"""
Training Logger with MLflow Integration

This module handles capturing training logs and sending them to MLflow
as artifacts when training completes or fails.
"""

import logging
import sys

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Optional
import traceback

class TrainingLogCapture:
    """Captures training logs and handles MLflow artifact upload."""

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        capture_stdout: bool = True,
        capture_stderr: bool = True,
    ):
        """
        Initialize training log capture.

        Args:
            experiment_name: MLflow experiment name
            run_name: MLflow run name
            capture_stdout: Whether to capture stdout
            capture_stderr: Whether to capture stderr
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.capture_stdout = capture_stdout
        self.capture_stderr = capture_stderr

        # Log storage
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.training_status = "unknown"
        self.error_info = None

        # Original streams
        self.original_stdout = None
        self.original_stderr = None

        # Log capture streams
        self.captured_stdout = None
        self.captured_stderr = None

        # MLflow tracking
        self.mlflow_run_id = None

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def start_capture(self):
        """Start capturing training logs."""
        self.start_time = datetime.now()
        self.training_status = "running"

        self.logger.info("Starting training log capture...")

        if self.capture_stdout:
            self.original_stdout = sys.stdout
            self.captured_stdout = StringIO()
            sys.stdout = TeeStream(self.original_stdout, self.captured_stdout)

        if self.capture_stderr:
            self.original_stderr = sys.stderr
            self.captured_stderr = StringIO()
            sys.stderr = TeeStream(self.original_stderr, self.captured_stderr)

        # Add log handler to capture logging messages
        self.log_handler = LogCaptureHandler(self.logs)
        logging.getLogger().addHandler(self.log_handler)

    def stop_capture(
        self, status: str = "completed", error_info: dict | None = None
    ):
        """
        Stop capturing logs and prepare for upload.

        Args:
            status: Training status (completed, failed, interrupted)
            error_info: Error information if training failed
        """
        self.end_time = datetime.now()
        self.training_status = status
        self.error_info = error_info

        self.logger.info(f"Stopping training log capture. Status: {status}")

        # Try to update MLflow run ID one last time before stopping
        # TODO: Implement update_mlflow_run_id() method
        # self.update_mlflow_run_id()

        # Restore original streams
        if self.capture_stdout and self.original_stdout:
            sys.stdout = self.original_stdout

        if self.capture_stderr and self.original_stderr:
            sys.stderr = self.original_stderr

        # Remove log handler
        if hasattr(self, "log_handler"):
            logging.getLogger().removeHandler(self.log_handler)

    def get_captured_logs(self) -> str:
        """Get all captured logs as a formatted string."""
        log_content = []

        # Add header information
        log_content.append("=" * 80)
        log_content.append(f"TRAINING LOG REPORT")
        log_content.append("=" * 80)
        log_content.append(f"Start Time: {self.start_time}")
        log_content.append(f"End Time: {self.end_time}")
        if (
            self.start_time
            and self.end_time
            and isinstance(self.start_time, datetime)
            and isinstance(self.end_time, datetime)
        ):
            duration = self.end_time - self.start_time
            log_content.append(f"Duration: {duration}")
        log_content.append(f"Status: {self.training_status.upper()}")
        log_content.append("")

        # Add error information if available
        if self.error_info:
            log_content.append("ERROR INFORMATION:")
            log_content.append("-" * 40)
            log_content.append(f"Error Type: {self.error_info.get('type', 'Unknown')}")
            log_content.append(
                f"Error Message: {self.error_info.get('message', 'No message')}"
            )
            if "traceback" in self.error_info:
                log_content.append("Traceback:")
                log_content.append(self.error_info["traceback"])
            log_content.append("")

        # Add captured stdout
        if self.captured_stdout:
            stdout_content = self.captured_stdout.getvalue()
            if stdout_content.strip():
                log_content.append("STDOUT CAPTURE:")
                log_content.append("-" * 40)
                log_content.append(stdout_content)
                log_content.append("")

        # Add captured stderr
        if self.captured_stderr:
            stderr_content = self.captured_stderr.getvalue()
            if stderr_content.strip():
                log_content.append("STDERR CAPTURE:")
                log_content.append("-" * 40)
                log_content.append(stderr_content)
                log_content.append("")

        # Add logged messages
        if self.logs:
            log_content.append("LOGGED MESSAGES:")
            log_content.append("-" * 40)
            for log_record in self.logs:
                timestamp = datetime.fromtimestamp(log_record["timestamp"])
                log_content.append(
                    f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{log_record['level']} - {log_record['logger']} - {log_record['message']}"
                )
            log_content.append("")

        log_content.append("=" * 80)
        log_content.append("END OF TRAINING LOG REPORT")
        log_content.append("=" * 80)

        return "\n".join(log_content)

    def save_logs_locally(
        self, output_dir: str = "./logs", log_filename: str | None = None
    ) -> str | None:
        """
        Save captured logs to local file.

        Args:
            output_dir: Directory to save logs
            log_filename: Custom filename for log file

        Returns:
            Path to saved log file, None if failed
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename
            if log_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"training_logs_{timestamp}_{self.training_status}.txt"

            log_file_path = output_path / log_filename

            # Write log content
            log_content = self.get_captured_logs()
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(log_content)

            self.logger.info(f"Training logs saved locally: {log_file_path}")
            return str(log_file_path)

        except Exception as e:
            self.logger.error(f"Failed to save logs locally: {e}")
            return None


class TeeStream:
    """Stream that writes to multiple outputs."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


class LogCaptureHandler(logging.Handler):
    """Custom logging handler that captures log records."""

    def __init__(self, log_storage):
        super().__init__()
        self.log_storage = log_storage

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            if record.exc_info:
                log_entry["exception"] = self.format(record)

            self.log_storage.append(log_entry)

        except Exception:
            self.handleError(record)


class TrainingLogContext:
    """Context manager for training log capture and upload."""

    def __init__(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        save_locally: bool = True,
        local_log_dir: str = "./logs",
    ):
        """
        Initialize training log context.

        Args:
            experiment_name: Experiment name
            run_name: Run name
            save_locally: Save logs locally
            local_log_dir: Local directory for log files
        """
        self.capture = TrainingLogCapture(experiment_name, run_name)
        self.save_locally = save_locally
        self.local_log_dir = local_log_dir

    def __enter__(self):
        self.capture.start_capture()
        self._capture = self.capture
        return self.capture

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.capture.stop_capture("completed")

        else:
            error_info = {
                "type": exc_type.__name__ if exc_type else "Unknown",
                "message": str(exc_val) if exc_val else "No message",
                "traceback": traceback.format_exc() if exc_tb else "No traceback",
            }

            self.capture.stop_capture("failed", error_info)

        if self.save_locally:
            self.capture.save_logs_locally(self.local_log_dir)

        return False
