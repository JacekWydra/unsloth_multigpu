"""
Custom Multi-GPU Trainer extending UnslothTrainer
Implements proper distributed training with gradient synchronization.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from unsloth import UnslothTrainer
import logging
from typing import Optional, Dict, Any, Union, List, Tuple, Callable
from torch.utils.data import Dataset, SequentialSampler
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.trainer_utils import EvalPrediction, has_length

logger = logging.getLogger(__name__)


class MultiGPUTrainer(UnslothTrainer):
    """Custom Multi-GPU Trainer that extends UnslothTrainer with proper distributed training.

    This trainer implements efficient multi-GPU training without the memory issues
    present in OpenSloth's implementation. It uses PyTorch's DistributedDataParallel
    for gradient synchronization and proper subprocess isolation for memory efficiency.

    Key improvements over OpenSloth:
    - No batch size multiplication (preserves configured batch size per device)
    - Proper gradient synchronization using PyTorch DDP with NCCL backend
    - Dynamic batch processing without memory scaling issues
    - Memory efficient data distribution via DistributedSampler
    - Process isolation to prevent CUDA context conflicts
    - Optimized synchronization only at gradient accumulation boundaries

    Attributes:
        rank: Process rank in distributed training (0 to world_size-1)
        local_rank: Local rank within the node
        world_size: Total number of processes in distributed training
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processing_class: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict[str, float]]] = None,
        args: Optional[TrainingArguments] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the CustomMultiGPUTrainer.

        Sets up distributed training environment, initializes the parent UnslothTrainer,
        and wraps the model with DistributedDataParallel for multi-GPU training.

        Args:
            model: The model to train
            processing_class: Tokenizer or processor for the model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            compute_metrics: Optional function to compute metrics during evaluation
            args: Training arguments
            **kwargs: Additional keyword arguments passed to parent trainer
        """

        # Get distributed training info first
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Initialize distributed training if not already done
        self._setup_distributed_training()

        logger.info(
            f"CustomMultiGPUTrainer init: rank={self.rank}, local_rank={self.local_rank}, world_size={self.world_size}"
        )

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            model = model.to(f"cuda:{self.local_rank}")

        # Initialize parent UnslothTrainer first (before DDP wrapping)
        super().__init__(
            model=model,
            processing_class=processing_class,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            args=args,
            **kwargs,
        )

        if self.world_size > 1:
            find_unused_parameters = hasattr(model, "peft_config") or hasattr(
                model, "base_model"
            )

            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=False,
            )
            logger.info(
                f"Wrapped model with DistributedDataParallel (find_unused_parameters={find_unused_parameters})"
            )

        # Disable progress bars on non-main processes to avoid duplicate tqdm displays
        if self.world_size > 1 and self.rank != 0:
            callbacks_to_remove = []
            for callback in self.callback_handler.callbacks:
                callback_name = str(type(callback).__name__).lower()
                if "progress" in callback_name or "tqdm" in callback_name:
                    callbacks_to_remove.append(callback)

            for callback in callbacks_to_remove:
                self.remove_callback(callback)

            # Set args to disable progress bars
            if hasattr(self.args, "disable_tqdm"):
                self.args.disable_tqdm = True

            logger.info(f"Rank {self.rank}: Disabled progress bars to avoid duplicates")

        logger.info(f"CustomMultiGPUTrainer initialized successfully")

    def _setup_distributed_training(self) -> None:
        """Initialize distributed training if not already initialized.

        Sets up the NCCL backend for distributed training if the world size
        is greater than 1 and distributed training hasn't been initialized yet.
        """
        if self.world_size > 1 and not dist.is_initialized():
            # Initialize distributed training
            dist.init_process_group(backend="nccl")
            logger.info(f"Initialized distributed training with NCCL backend")

    def _get_train_sampler(
        self, train_dataset: Optional[Dataset] = None
    ) -> Optional[DistributedSampler]:
        """Override to control progress display.

        Returns:
            Training sampler (typically DistributedSampler for multi-GPU)
        """
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None
        if self.world_size > 1:
            logger.info(
                f"Using DistributedSampler: rank={self.rank}, world_size={self.world_size}, drop_last=True"
            )
            return DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.args.seed,
                drop_last=True,
            )
        return None

    def _get_eval_sampler(
        self, eval_dataset: Optional[Dataset] = None
    ) -> Optional[DistributedSampler]:
        if eval_dataset is None or not has_length(eval_dataset):
            return None
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            logger.info(
                f"Using DistributedSampler: rank={self.rank}, world_size={self.world_size}, drop_last=True"
            )
            return DistributedSampler(
                eval_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=self.args.seed,
                drop_last=True,  # CRITICAL: Ensure all GPUs get same number of batches
            )
        return None

    def get_train_dataloader(self) -> DataLoader:
        """Create train dataloader with proper data distribution across GPUs.

        Key improvements over standard implementations:
        - No batch size multiplication by world size
        - Proper data splitting using DistributedSampler
        - Ensures all GPUs get the same number of batches via drop_last=True

        Returns:
            DataLoader configured for distributed training

        Raises:
            ValueError: If train_dataset is None
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self.args.per_device_train_batch_size,  # No multiplication!
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """Create eval dataloader with proper data distribution.

        Args:
            eval_dataset: Optional evaluation dataset. If None, uses self.eval_dataset

        Returns:
            DataLoader configured for distributed evaluation

        Raises:
            ValueError: If both eval_dataset and self.eval_dataset are None
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        return self._get_dataloader(
            dataset=eval_dataset,
            description="Evaluation",
            batch_size=self.args.per_device_eval_batch_size,  # No multiplication!
            sampler_fn=self._get_eval_sampler,
            is_training=False,
        )

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ) -> None:
        """Save model only on rank 0 to prevent conflicts.

        In distributed training, only the main process (rank 0) should save
        the model to avoid race conditions and duplicate saves.

        Args:
            output_dir: Directory to save the model. If None, uses default
            _internal_call: Whether this is an internal call from the trainer
        """
        if self.world_size > 1 and self.rank != 0:
            # Only rank 0 saves the model
            return

        # For DDP, access the underlying model
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # Temporarily replace self.model for saving
        original_model = self.model
        self.model = model_to_save

        try:
            super().save_model(output_dir, _internal_call)
        finally:
            self.model = original_model

        logger.info(f"Model saved by rank {self.rank}")

    def _gather_and_numpify(self, tensors: List[torch.Tensor]) -> List[Any]:
        """Override to handle distributed evaluation properly.

        Gathers tensors from all processes during distributed evaluation
        to ensure metrics are computed over the complete dataset.

        Args:
            tensors: List of tensors to gather from all processes

        Returns:
            List of gathered and converted tensors
        """
        if self.world_size <= 1:
            return super()._gather_and_numpify(tensors)

        # Gather tensors from all processes
        gathered_tensors = []
        for tensor in tensors:
            if tensor is None:
                continue

            # Gather from all processes
            gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered, tensor)
            gathered_tensors.extend(gathered)

        return super()._gather_and_numpify(gathered_tensors)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Override compute_loss to properly scale loss for multi-GPU gradient accumulation.
        """

        if "labels" in inputs and inputs["labels"] is not None:
            labels = inputs["labels"]
            valid_tokens = (labels != -100).sum().item()
            total_tokens = labels.numel()
            valid_ratio = valid_tokens / total_tokens if total_tokens > 0 else 0.0

            logger.debug(
                f"Rank {self.rank}: Label validation - Valid tokens: {valid_tokens}/{total_tokens} ({valid_ratio:.3f})"
            )

            if valid_tokens == 0:
                logger.warning(
                    f"Rank {self.rank}: All labels are masked (-100)! Skipping loss computation to prevent NaN"
                )
                logger.debug(f"Rank {self.rank}: Labels shape: {labels.shape}")

                self._store_all_masked_batch_example(inputs)

                # Return zero loss to prevent NaN - this batch contributes nothing to learning
                # In distributed training, other GPUs will have valid batches to learn from
                dummy_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)

                if return_outputs:
                    # Create dummy outputs for consistency
                    dummy_outputs = type("DummyOutputs", (), {})()
                    dummy_outputs.logits = torch.zeros(
                        labels.shape + (getattr(model.config, "vocab_size", 32000),),
                        device=labels.device,
                    )
                    return dummy_loss, dummy_outputs
                return dummy_loss

            elif valid_ratio < 0.01:  # Less than 1% valid tokens
                logger.warning(
                    f"Rank {self.rank}: Very few valid tokens ({valid_ratio:.3f}) - may cause numerical instability"
                )

            # Log some label statistics for debugging
            if self.rank == 0 and hasattr(self, "_label_debug_step_count"):
                self._label_debug_step_count += 1
            elif self.rank == 0:
                self._label_debug_step_count = 0

            if self.rank == 0 and self._label_debug_step_count % 10 == 0:
                logger.info(
                    f"Step {self._label_debug_step_count}: Batch label stats - Valid: {valid_tokens}/{total_tokens} ({valid_ratio:.3f})"
                )

        # Get loss from parent
        if return_outputs:
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            loss = super().compute_loss(
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=num_items_in_batch,
            )

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Rank {self.rank}: CRITICAL - Loss is {loss.item()}!")
            if "labels" in inputs and inputs["labels"] is not None:
                labels = inputs["labels"]
                valid_tokens = (labels != -100).sum().item()
                logger.error(
                    f"Rank {self.rank}: Associated labels had {valid_tokens} valid tokens out of {labels.numel()}"
                )
                logger.error(
                    f"Rank {self.rank}: Labels sample: {labels.flatten()[:20].tolist()}"
                )
            # Log model outputs if available
            if return_outputs and "outputs" in locals():
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    logger.error(
                        f"Rank {self.rank}: Logits shape: {logits.shape}, Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]"
                    )
                    logger.error(
                        f"Rank {self.rank}: Logits has NaN: {torch.isnan(logits).any().item()}, Logits has Inf: {torch.isinf(logits).any().item()}"
                    )

        # Manual scaling for multi-GPU - empirically necessary for correct gradients
        if self.world_size > 1:
            loss = loss / self.world_size
            logger.debug(
                f"Rank {self.rank}: Scaled loss by world_size={self.world_size}"
            )

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Minimal override to ensure proper DDP gradient synchronization with accumulation.

        The key issue is that with DDP, gradients are automatically averaged across GPUs
        after each backward pass. With gradient accumulation, we want to:
        1. Accumulate gradients locally for non-update steps
        2. Only sync and average gradients on update steps
        """
        # Track which step we're on within the accumulation cycle
        if not hasattr(self, "_step_count"):
            self._step_count = 0

        current_step_in_accumulation = (
            self._step_count % self.args.gradient_accumulation_steps
        )
        is_last_accumulation_step = (
            current_step_in_accumulation == self.args.gradient_accumulation_steps - 1
        )

        if self.world_size > 1 and hasattr(model, "no_sync"):
            # For all steps except the last in accumulation cycle, disable gradient sync
            if not is_last_accumulation_step:
                with model.no_sync():
                    # Call parent's training_step with gradient sync disabled
                    loss = super().training_step(model, inputs, num_items_in_batch)
            else:
                # Last accumulation step - allow gradient sync
                loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            # Single GPU or no DDP - just call parent
            loss = super().training_step(model, inputs, num_items_in_batch)

        self._step_count += 1
        return loss

    def _store_all_masked_batch_example(self, inputs: Dict[str, torch.Tensor]) -> None:
        """Store examples of all-masked batches for dataset verification.

        This method stores the tokenized text examples when all labels are -100
        to help verify if this is a dataset issue or a processing issue.

        Args:
            inputs: The batch inputs containing input_ids, labels, etc.
        """
        import os
        import json
        from datetime import datetime

        # Initialize storage if not exists
        if not hasattr(self, "_all_masked_examples_count"):
            self._all_masked_examples_count = 0

        self._all_masked_examples_count += 1

        # Only store first few examples to avoid excessive disk usage
        if self._all_masked_examples_count > 5:
            return

        try:
            # Create debug directory if it doesn't exist
            debug_dir = "debug/all_masked_batches"
            os.makedirs(debug_dir, exist_ok=True)

            # Prepare data for storage
            batch_data = {
                "timestamp": datetime.now().isoformat(),
                "rank": self.rank,
                "world_size": self.world_size,
                "example_number": self._all_masked_examples_count,
                "batch_size": inputs["input_ids"].shape[0],
                "sequence_length": inputs["input_ids"].shape[1],
                "examples": [],
            }

            # Process each example in the batch
            for i in range(inputs["input_ids"].shape[0]):
                input_ids = inputs["input_ids"][i].cpu().tolist()
                labels = inputs["labels"][i].cpu().tolist()

                example_data = {
                    "sequence_index": i,
                    "input_ids": input_ids,
                    "labels": labels,
                    "input_ids_length": len(input_ids),
                    "labels_length": len(labels),
                    "valid_label_positions": [
                        j for j, label in enumerate(labels) if label != -100
                    ],
                    "total_valid_labels": sum(1 for label in labels if label != -100),
                }

                # Try to decode the text if we have access to a tokenizer
                if (
                    hasattr(self, "processing_class")
                    and self.processing_class is not None
                ):
                    try:
                        # Decode the full text
                        full_text = self.processing_class.decode(
                            input_ids, skip_special_tokens=False
                        )
                        example_data["decoded_text"] = full_text

                        # Try to decode just the response part (where labels != -100)
                        response_ids = [
                            input_ids[j]
                            for j in range(len(labels))
                            if labels[j] != -100
                        ]
                        if response_ids:
                            response_text = self.processing_class.decode(
                                response_ids, skip_special_tokens=False
                            )
                            example_data["response_part"] = response_text
                        else:
                            example_data["response_part"] = "NO_RESPONSE_TOKENS"

                    except Exception as decode_error:
                        example_data["decode_error"] = str(decode_error)
                        logger.debug(f"Could not decode example {i}: {decode_error}")

                batch_data["examples"].append(example_data)

            # Save to file
            filename = f"all_masked_batch_rank_{self.rank}_example_{self._all_masked_examples_count}.json"
            filepath = os.path.join(debug_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Rank {self.rank}: Stored all-masked batch example #{self._all_masked_examples_count} to {filepath}"
            )

            # Log summary to console
            logger.warning(
                f"Rank {self.rank}: ALL-MASKED BATCH EXAMPLE #{self._all_masked_examples_count}"
            )
            logger.warning(
                f"Rank {self.rank}: Batch size: {batch_data['batch_size']}, Sequence length: {batch_data['sequence_length']}"
            )

            if batch_data["examples"]:
                first_example = batch_data["examples"][0]
                if "decoded_text" in first_example:
                    # Show first 200 chars of the decoded text
                    text_preview = (
                        first_example["decoded_text"][:200] + "..."
                        if len(first_example["decoded_text"]) > 200
                        else first_example["decoded_text"]
                    )
                    logger.warning(
                        f"Rank {self.rank}: First example text: {repr(text_preview)}"
                    )
                if "response_part" in first_example:
                    logger.warning(
                        f"Rank {self.rank}: Response part: {repr(first_example['response_part'])}"
                    )

        except Exception as e:
            logger.error(
                f"Rank {self.rank}: Error storing all-masked batch example: {e}"
            )

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Log only from rank 0 to avoid duplicate logs.

        In distributed training, only the main process should log metrics
        to avoid duplicate entries in logging systems like MLflow.

        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for timing calculations
        """
        if self.world_size > 1 and self.rank != 0:
            return

        # Let HF Trainer's logging logic handle frequency based on logging_steps
        # The parent trainer already respects self.control.should_log
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

    def _maybe_log_save_evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Override to control evaluation progress display.

        Only the main process performs logging, saving, and evaluation
        to avoid duplicate operations in distributed training.

        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments

        Returns:
            Evaluation results from parent method (only on rank 0)
        """
        if self.world_size > 1 and self.rank != 0:
            # Non-main processes: skip logging/evaluation to avoid progress bars
            return

        return super()._maybe_log_save_evaluate(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, Any]:
        """Override evaluation loop to disable progress bars on non-main processes.

        Args:
            dataloader: DataLoader for evaluation data
            description: Description for the evaluation loop
            prediction_loss_only: Whether to only compute prediction loss
            ignore_keys: Keys to ignore in evaluation outputs
            metric_key_prefix: Prefix for metric names

        Returns:
            Dictionary containing evaluation results
        """
        if self.world_size > 1 and self.rank != 0:
            # For non-main processes, run evaluation without progress display
            import transformers

            original_disable_tqdm = getattr(self.args, "disable_tqdm", False)
            self.args.disable_tqdm = True

            try:
                result = super().evaluation_loop(
                    dataloader,
                    description,
                    prediction_loss_only,
                    ignore_keys,
                    metric_key_prefix,
                )
            finally:
                self.args.disable_tqdm = original_disable_tqdm

            return result
        else:
            # Main process can show progress bars
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix,
            )

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Optional[Any] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Override train method to control progress bars per process.

        Non-main processes have progress bars disabled to avoid cluttered output.
        Only the main process (rank 0) displays training progress.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            trial: Optuna trial object for hyperparameter optimization
            ignore_keys_for_eval: Keys to ignore during evaluation
            **kwargs: Additional keyword arguments

        Returns:
            Training results
        """
        if self.world_size > 1 and self.rank != 0:
            # For non-main processes, disable all progress bars
            original_disable_tqdm = getattr(self.args, "disable_tqdm", False)
            self.args.disable_tqdm = True

            # Also set environment variable to disable tqdm completely
            import os

            original_tqdm_disable = os.environ.get("TQDM_DISABLE", "0")
            os.environ["TQDM_DISABLE"] = "1"

            try:
                logger.info(f"Rank {self.rank}: Training with progress bars disabled")
                result = super().train(
                    resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs
                )
            finally:
                # Restore original settings
                self.args.disable_tqdm = original_disable_tqdm
                os.environ["TQDM_DISABLE"] = original_tqdm_disable

            return result
        else:
            # Main process can show progress bars
            logger.info(
                f"Rank {self.rank}: Training with progress bars enabled (main process)"
            )
            return super().train(
                resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs
            )

    def _memory_tracker_report(self) -> None:
        """Report memory usage per GPU.

        Logs current GPU memory allocation and reservation for monitoring
        memory usage during training.
        """
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU {self.local_rank} memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
            )

    def create_model_card(self, *args: Any, **kwargs: Any) -> Any:
        """Override to handle DDP model access for model card creation.

        When using DistributedDataParallel, we need to access the underlying
        model for operations like model card creation.

        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments

        Returns:
            Model card creation result
        """
        # For DDP, access the underlying model
        original_model = self.model
        if hasattr(self.model, "module"):
            self.model = self.model.module

        try:
            return super().create_model_card(*args, **kwargs)
        finally:
            self.model = original_model

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        trial: Optional[Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> Any:
        """Override to handle DDP model access during checkpoint saving.

        When using DistributedDataParallel, we need to access the underlying
        model for checkpoint saving operations.

        Args:
            model: Model to save checkpoint for
            trial: Optuna trial object
            metrics: Optional metrics dictionary

        Returns:
            Checkpoint saving result
        """
        # For DDP, access the underlying model
        original_model = self.model
        if hasattr(self.model, "module"):
            self.model = self.model.module

        try:
            return super()._save_checkpoint(model, trial)
        finally:
            self.model = original_model


def setup_distributed_environment(rank: int, world_size: int, master_port: int) -> None:
    """Setup environment variables for distributed training.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        master_port: Port for inter-process communication
    """
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)  # Assuming one GPU per process
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    # Ensure proper CUDA device visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    logger.info(
        f"Setup distributed environment: rank={rank}, world_size={world_size}, port={master_port}"
    )


def cleanup_distributed_training() -> None:
    """Cleanup distributed training.

    Destroys the distributed process group if it was initialized.
    This should be called when training is complete or interrupted.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed distributed process group")
