#!/usr/bin/env python3
"""
Custom worker process for distributed training using MultiGPUTrainer.

This module implements isolated worker processes for multi-GPU training.
Each worker process handles training on a single GPU with proper CUDA isolation.
"""

import os
import sys
import logging

# Add current working directory to Python path for module imports
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def setup_distributed_environment(
    gpu_id: int, rank: int, world_size: int, master_port: int
) -> None:
    """Set up distributed training environment for this worker process.

    This function configures the environment variables needed for PyTorch
    distributed training and sets CUDA device visibility for proper isolation.

    Args:
        gpu_id: GPU device ID this worker should use
        rank: Process rank in distributed training (0 to world_size-1)
        world_size: Total number of processes in distributed training
        master_port: Port for inter-process communication
    """
    # Set CUDA_VISIBLE_DEVICES before any CUDA imports
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set up distributed training environment variables
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"  # Always 0 since each process sees one GPU
    os.environ["WORLD_SIZE"] = str(world_size)
    # if world_size > 1 and master_port > 0:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    print(f"Worker process {os.getpid()} starting on GPU {gpu_id}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"RANK: {rank}, LOCAL_RANK: 0, WORLD_SIZE: {world_size}")
    print(f"MASTER_ADDR: localhost, MASTER_PORT: {master_port}")


def run_custom_training(gpu_id: int, rank: int, world_size: int) -> bool:
    """Run training using our custom multi-GPU trainer.

    This function executes the complete training pipeline for a single GPU worker:
    - Sets up CUDA environment and verifies device isolation
    - Loads model and dataset with caching support
    - Creates CustomMultiGPUTrainer with proper distributed configuration
    - Executes training with memory and batch size monitoring

    Args:
        gpu_id: GPU device ID for this worker
        rank: Process rank in distributed training
        world_size: Total number of processes
        config_data: Training configuration dictionary
        training_args_data: Training arguments dictionary

    Returns:
        True if training completed successfully, False otherwise
    """

    # Now import CUDA-dependent modules
    import os

    import unsloth
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from transformers import AutoTokenizer


    from unsloth_multigpu.multigpu_trainer import MultiGPUTrainer
    from unsloth_multigpu.metrics.metrics import MetricsComputer
    from unsloth_multigpu.utils import load_training_config
    from unsloth_multigpu.data_model.data_model import TrainingConfig
    from unsloth_multigpu.dataset.select_dataset import get_training_dataset
    
    def _load_dataset(config: TrainingConfig, tokenizer: AutoTokenizer) -> Dataset:
        logger.info("Loading training dataset...")
        dataset = get_training_dataset(config.dataset_config, tokenizer)
        return dataset["train"], dataset["validation"]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"=== CUSTOM WORKER PROCESS {rank} STARTING ON GPU {gpu_id} ===")
    logger.info(f"Process PID: {os.getpid()}")

    # Verify CUDA setup
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        logger.info(f"CUDA device count: {device_count}")
        logger.info(f"Current device: {current_device}")
        logger.info(f"Device name: {device_name}")

        if device_count == 1:
            logger.info("✅ Proper device isolation achieved!")
        else:
            logger.warning(
                f"⚠ Expected 1 device, got {device_count}. Device isolation may have failed."
            )

        # Log memory state
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(
            f"Initial memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB"
        )
    else:
        logger.error("CUDA not available in worker process!")
        return False

    try:
        # Create TrainingConfig from config data
        training_config = load_training_config(
            os.environ.get("CONFIG_PATH", "/configs/config.yaml")
        )

        logger.info("=== PREPARING MODEL AND TOKENIZER ===")

        # Load model and tokenizer using the same approach as single-GPU training
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=training_config.pretrained_model_config.model_name,
            max_seq_length=training_config.pretrained_model_config.max_seq_length,
            dtype=training_config.pretrained_model_config.dtype,
            load_in_4bit=training_config.pretrained_model_config.load_in_4bit,
        )

        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=training_config.peft_config.r,
            target_modules=training_config.peft_config.target_modules,
            lora_alpha=training_config.peft_config.lora_alpha,
            lora_dropout=training_config.peft_config.lora_dropout,
            bias=training_config.peft_config.bias,
            use_gradient_checkpointing=training_config.peft_config.use_gradient_checkpointing,
            random_state=training_config.peft_config.random_state,
            use_rslora=training_config.peft_config.use_rslora,
        )

        logger.info("=== LOADING DATASET ===")

        train_dataset, eval_dataset = _load_dataset(training_config, tokenizer)

        metrics_fn = MetricsComputer(tokenizer=tokenizer)

        if world_size > 1:
            # Multi-GPU: Only rank 0 should have logging enabled
            if rank != 0:
                logger.info(
                    f"Rank {rank}: Disabling logging reporting to prevent duplicate runs"
                )
                original_report_to = training_config.trainer_config.report_to
                training_config.trainer_config.report_to = []
                logger.info(
                    f"Rank {rank}: Changed report_to from {original_report_to} to [] (empty)"
                )

        logger.info("=== BATCH SIZE VERIFICATION ===")
        logger.info(
            f"Config per_device_train_batch_size: {training_config.trainer_config.per_device_train_batch_size}"
        )
        logger.info(
            f"Config gradient_accumulation_steps: {training_config.trainer_config.gradient_accumulation_steps}"
        )
        logger.info(f"World size: {world_size}")
        logger.info(
            f"Expected effective batch size: {training_config.trainer_config.per_device_train_batch_size * world_size * training_config.trainer_config.gradient_accumulation_steps}"
        )

        logger.info("=== CREATING CUSTOM MULTI-GPU TRAINER ===")


        logger.info(
            f"Using original datasets - Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples"
        )

        # Debug: Log dataset types and sample structure
        logger.info(f"Train dataset type: {type(train_dataset)}")
        logger.info(f"Eval dataset type: {type(eval_dataset)}")
        if hasattr(train_dataset, "column_names"):
            logger.info(f"Train dataset columns: {train_dataset.column_names}")
        if hasattr(eval_dataset, "column_names"):
            logger.info(f"Eval dataset columns: {eval_dataset.column_names}")

        # Debug: Check first sample structure
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info(f"First train sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                logger.info(
                    f"  {key}: {type(value)} - length: {len(value) if isinstance(value, (list, str)) else 'N/A'}"
                )

        # Handle empty evaluation datasets (can happen in distributed caching)
        if eval_dataset is not None and len(eval_dataset) == 0:
            logger.warning(
                f"Rank {rank} received empty eval dataset, setting to None to avoid trainer errors"
            )
            eval_dataset = None

        # Create trainer based on world size
        if world_size > 1:
            # Multi-GPU: Use custom distributed trainer
            from unsloth_multigpu.multigpu_trainer import MultiGPUTrainer
            training_config.trainer_config.drop_last = True

            trainer = MultiGPUTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metrics_fn,
                args=training_config.trainer_config,
            )
            logger.info("Created MultiGPUTrainer for distributed training")
        else:
            # Single GPU: Use regular UnslothTrainer
            from unsloth import UnslothTrainer

            trainer = UnslothTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metrics_fn,
                args=training_config.trainer_config,
            )
            logger.info("Created UnslothTrainer for single GPU training")

        # Apply train_on_responses_only if needed
        if training_config.train_on_responses:
            logger.info("=== APPLYING train_on_responses_only ===")
            # Debug: Log dataset structure before train_on_responses_only
            logger.info(
                f"Before train_on_responses_only - Train dataset type: {type(trainer.train_dataset)}"
            )
            logger.info(
                f"Before train_on_responses_only - Columns: {trainer.train_dataset.column_names}"
            )
            if len(trainer.train_dataset) > 0:
                sample = trainer.train_dataset[0]
                logger.info(
                    f"Before train_on_responses_only - Sample keys: {list(sample.keys())}"
                )

            trainer = train_on_responses_only(
                trainer=trainer,
                instruction_part=training_config.dataset_config.instruction_part,
                response_part=training_config.dataset_config.response_part,
            )
            # Debug: Log dataset structure after train_on_responses_only
            logger.info(
                f"After train_on_responses_only - Train dataset type: {type(trainer.train_dataset)}"
            )
            logger.info(
                f"After train_on_responses_only - Columns: {trainer.train_dataset.column_names}"
            )
            if len(trainer.train_dataset) > 0:
                sample = trainer.train_dataset[0]
                logger.info(
                    f"After train_on_responses_only - Sample keys: {list(sample.keys())}"
                )
                for key, value in sample.items():
                    if key == "labels" and isinstance(value, list):
                        logger.info(
                            f"  {key}: {type(value)} - length: {len(value)} - sample: {value[:10]}...{value[-10:]}"
                        )
                    else:
                        logger.info(
                            f"  {key}: {type(value)} - length: {len(value) if isinstance(value, (list, str)) else 'N/A'}"
                        )


        # Log memory after model loading
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(
            f"After model load - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB"
        )

        # Test dataloader to verify batch size
        logger.info("=== VERIFYING DATALOADER BATCH SIZE ===")
        train_dataloader = trainer.get_train_dataloader()
        sample_batch = next(iter(train_dataloader))
        if isinstance(sample_batch, dict) and "input_ids" in sample_batch:
            actual_batch_size = sample_batch["input_ids"].shape[0]
            logger.info(
                f"✅ ACTUAL BATCH SIZE IN MODEL FORWARD PASS: {actual_batch_size}"
            )
            logger.info(f"✅ Sequence length: {sample_batch['input_ids'].shape[1]}")

            if actual_batch_size == training_config.trainer_config.per_device_train_batch_size:
                logger.info(
                    "✅ BATCH SIZE CORRECT: Matches config without multiplication!"
                )
            else:
                logger.error(
                    f"❌ BATCH SIZE MISMATCH: Expected {training_config.trainer_config.per_device_train_batch_size}, got {actual_batch_size}"
                )

        logger.info("=== STARTING TRAINING ===")

        # Start training
        result = trainer.train()

        # Final memory report
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(
            f"Training completed - Final memory: Allocated {memory_allocated:.2f} GB, Reserved {memory_reserved:.2f} GB"
        )

        logger.info(
            f"✅ Custom multi-GPU training completed successfully on GPU {gpu_id}"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Training failed on GPU {gpu_id}: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: custom_worker_process.py <gpu_id> <rank> <world_size> <master_port>"
        )
        sys.exit(1)

    gpu_id: int = int(sys.argv[1])
    rank: int = int(sys.argv[2])
    world_size: int = int(sys.argv[3])
    master_port_str = sys.argv[4]
    master_port: int = int(master_port_str) if master_port_str != "None" else 0

    # Setup distributed environment first
    setup_distributed_environment(gpu_id, rank, world_size, master_port)

    # Run training
    success: bool = run_custom_training(gpu_id, rank, world_size)

    # Exit with appropriate code
    sys.exit(0 if success else 1)
