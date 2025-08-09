#!/usr/bin/env python3
"""
Main training script using our custom multi-GPU implementation.

This script provides the entry point for multi-GPU training using our custom
implementation that avoids OpenSloth's memory multiplication issues.
It handles configuration loading, environment setup, and training orchestration.
"""
import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from types import FrameType
from typing import Any, Dict, List, Optional, Tuple

from unsloth_multigpu.data_model.data_model import TrainingConfig
from unsloth_multigpu.logger.training_logger import TrainingLogContext
from unsloth_multigpu.utils import load_training_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_training_configuration(config: TrainingConfig, devices: List[int]) -> None:
    """Log training configuration details.

    Args:
        config: Training configuration
        devices: List of GPU device IDs
    """
    logger.info("=== TRAINING CONFIGURATION ===")
    logger.info(f"Model: {config.pretrained_model_config.model_name}")
    logger.info(f"Max sequence length: {config.pretrained_model_config.max_seq_length}")
    logger.info(
        f"Batch size per device: {config.trainer_config.per_device_train_batch_size}"
    )
    logger.info(
        f"Gradient accumulation steps: {config.trainer_config.gradient_accumulation_steps}"
    )
    logger.info(
        f"Total effective batch size: {config.trainer_config.per_device_train_batch_size * len(devices) * config.trainer_config.gradient_accumulation_steps}"
    )
    logger.info(f"Learning rate: {config.trainer_config.learning_rate}")
    logger.info(f"Number of epochs: {config.trainer_config.num_train_epochs}")


def main() -> Optional[Any]:
    """Main training function using custom multi-GPU implementation.

    Orchestrates the complete training pipeline:
    - Loads configuration from environment or default path
    - Sets up devices and training parameters
    - Configures MLflow logging if enabled
    - Executes distributed training with log capture

    Returns:
        Training result if successful, None if failed

    Raises:
        Various exceptions from configuration loading or training execution
    """

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
        """Handle interrupt signals for graceful shutdown.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info("🛑 Received interrupt signal, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=== STARTING CUSTOM MULTI-GPU TRAINING ===")

    # Load configuration
    config_path = os.environ.get("CONFIG_PATH", "/configs/config.yaml")
    config = load_training_config(config_path)

    # Get devices from config
    devices = getattr(config, "devices", [0])
    if not devices:
        devices = [0]  # Default to single GPU

    logger.info(f"Training will use {len(devices)} GPUs: {devices}")

    # Log training configuration
    log_training_configuration(config, devices)

    with TrainingLogContext(
        experiment_name=os.environ.get(
            "EXPERIMENT_NAME", "llm_finetune_causal_reports"
        ),
        run_name=getattr(config.trainer_config, "run_name", "multi_gpu_training"),
        save_locally=True,
        local_log_dir="./logs",
    ) as log_capture:
        try:
            # Run custom multi-GPU training
            logger.info("Starting custom multi-GPU training...")
            logger.info(
                f"Starting CUSTOM multi-GPU training on {len(devices)} GPUs: {devices}"
            )


            if len(devices) == 1:
                logger.info("Single GPU training with custom trainer")
                master_port = 0
                
            else:
                # Multi-GPU training with subprocess isolation
                # Find a free port for multi-GPU coordination
                master_port = find_free_port()
                logger.info(f"Using port {master_port} for distributed training")
                logger.info("Starting CUSTOM multi-GPU training with subprocess isolation")


            processes = []
            # Start a subprocess for each GPU
            for rank, gpu_id in enumerate(devices):
                logger.info(
                    f"Starting CUSTOM subprocess for GPU {gpu_id} (rank {rank})"
                )

                process = start_custom_worker_subprocess(
                    gpu_id, rank, len(devices), master_port
                )
                processes.append((gpu_id, process))

            # Wait for all processes to complete
            monitor_processes(processes)

            logger.info("✅ Training completed successfully!")
            logger.info("Training logs will be uploaded to MLflow as artifacts")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info(
                "Training logs (including error details) will be uploaded to MLflow"
            )
            raise
    return None

def find_free_port() -> int:
    """Find a free port for distributed training.

    Returns:
        An available port number for network communication.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def start_custom_worker_subprocess(
    gpu_id: int, rank: int, world_size: int, master_port: int
) -> subprocess.Popen:
    """Start a custom worker subprocess for training on a specific GPU.

    Args:
        gpu_id: GPU device ID to use for this worker
        rank: Process rank in distributed training (0 to world_size-1)
        world_size: Total number of processes in distributed training
        master_port: Port for inter-process communication
        config_file: Path to pickled configuration file
        args_file: Path to pickled training arguments file

    Returns:
        Subprocess.Popen object for the started worker process
    """
    worker_script = os.path.join(os.path.dirname(__file__), "worker_process.py")

    cmd = [
        sys.executable,
        worker_script,
        str(gpu_id),
        str(rank),
        str(world_size),
        str(master_port),
    ]

    logger.info(f"Starting CUSTOM worker subprocess: {' '.join(cmd)}")

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )


def monitor_processes(processes: List[Tuple[int, subprocess.Popen]]) -> None:
    """Monitor all processes with timeout and immediate failure detection.

    This function monitors multiple GPU training processes, handling:
    - Real-time output streaming from all processes
    - Graceful shutdown on SIGINT/SIGTERM signals
    - Timeout detection and process cleanup
    - Immediate failure detection and cascading shutdown

    Args:
        processes: List of (gpu_id, subprocess.Popen) tuples to monitor

    Raises:
        RuntimeError: If any process fails or training times out
    """

    # Create threads to monitor each process
    process_results: Dict[int, int] = {}
    process_threads: List[Tuple[int, threading.Thread, subprocess.Popen]] = []

    # Global flag for graceful shutdown
    shutdown_requested: bool = False

    def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
        """Handle SIGINT (Ctrl+C) and SIGTERM signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            logger.info(
                "🛑 SIGNAL RECEIVED: Gracefully shutting down all training processes..."
            )

            # Terminate all processes
            for gpu_id, thread, process in process_threads:
                if process.poll() is None:
                    logger.info(f"Terminating CUSTOM GPU {gpu_id} process")
                    try:
                        process.terminate()
                    except Exception as e:
                        logger.error(f"Error terminating GPU {gpu_id} process: {e}")

            # Wait a bit for graceful shutdown
            time.sleep(3)

            # Force kill if needed
            for gpu_id, thread, process in process_threads:
                if process.poll() is None:
                    logger.warning(f"Force killing CUSTOM GPU {gpu_id} process")
                    try:
                        process.kill()
                    except Exception as e:
                        logger.error(f"Error killing GPU {gpu_id} process: {e}")

            logger.info("✅ All training processes terminated")
            sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Register cleanup function for process termination
    def cleanup_processes() -> None:
        """Clean up all processes on exit.

        This function is registered with atexit to ensure all subprocesses
        are properly terminated when the main process exits.
        """
        for gpu_id, thread, process in process_threads:
            if process.poll() is None:
                try:
                    process.terminate()
                    time.sleep(1)
                    if process.poll() is None:
                        process.kill()
                except Exception:
                    pass

    atexit.register(cleanup_processes)

    def monitor_process(gpu_id: int, process: subprocess.Popen) -> None:
        """Monitor a single process and capture its output.

        This function runs in a separate thread to monitor each GPU process,
        streaming its output in real-time and recording the exit code.

        Args:
            gpu_id: GPU ID being monitored
            process: Subprocess to monitor
        """
        try:
            logger.info(f"[CUSTOM GPU {gpu_id}] Process monitoring started")

            # Stream output in real-time
            while True:
                try:
                    line = process.stdout.readline()
                    if not line:  # Process finished
                        break
                    print(f"[CUSTOM GPU {gpu_id}] {line.rstrip()}")
                except Exception as e:
                    logger.error(f"[CUSTOM GPU {gpu_id}] Error reading output: {e}")
                    break

            # Wait for process to complete
            try:
                exit_code = process.wait(timeout=30)  # 30 second timeout for cleanup
                process_results[gpu_id] = exit_code
                logger.info(
                    f"[CUSTOM GPU {gpu_id}] Process completed with exit code {exit_code}"
                )
            except subprocess.TimeoutExpired:
                logger.error(f"[CUSTOM GPU {gpu_id}] Process cleanup timeout")
                process.kill()
                process_results[gpu_id] = -1

        except Exception as e:
            logger.error(f"[CUSTOM GPU {gpu_id}] Process monitoring failed: {e}")
            process_results[gpu_id] = -1

    # Start monitoring threads
    for gpu_id, process in processes:
        thread = threading.Thread(target=monitor_process, args=(gpu_id, process))
        thread.daemon = True
        thread.start()
        process_threads.append((gpu_id, thread, process))

    # Monitor all processes with periodic checks
    start_time = time.time()
    check_interval = 30  # Check every 30 seconds

    while True:
        # Check for shutdown signal
        if shutdown_requested:
            logger.info("Shutdown requested, exiting monitoring loop")
            break

        # Check timeout
        elapsed_hours = (time.time() - start_time) / 3600

        # Check if all processes are done
        all_done = True
        for gpu_id, thread, process in process_threads:
            if thread.is_alive() or process.poll() is None:
                all_done = False
                break

        if all_done:
            break

        # Check if any process has failed
        failed_processes = []
        for gpu_id in process_results:
            if process_results[gpu_id] != 0:
                failed_processes.append(gpu_id)

        if failed_processes:
            logger.error(
                f"CUSTOM GPU processes {failed_processes} have failed. Terminating all processes."
            )
            # Kill all remaining processes
            for gpu_id, thread, process in process_threads:
                if process.poll() is None:
                    logger.warning(
                        f"Terminating CUSTOM GPU {gpu_id} process due to failure in other processes"
                    )
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
             

        # Wait before next check
        time.sleep(check_interval)
        logger.info(
            f"CUSTOM training progress check: {elapsed_hours:.1f}h elapsed, {len(process_results)}/{len(processes)} processes completed"
        )

    # Final check of all results
    failed_processes = []
    for gpu_id, exit_code in process_results.items():
        if exit_code != 0:
            failed_processes.append(gpu_id)

    if failed_processes:
        raise RuntimeError(f"CUSTOM training failed on GPU(s): {failed_processes}")

    logger.info("All CUSTOM GPU subprocesses completed successfully")
    return None


if __name__ == "__main__":
    main()
