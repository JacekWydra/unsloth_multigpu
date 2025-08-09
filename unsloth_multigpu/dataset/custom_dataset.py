"""Custom dataset implementation template.

Users should implement the get_dataset() function to load their own datasets.
"""

import json
import logging
from pathlib import Path
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from unsloth_multigpu.data_model.data_model import DatasetConfig

logger = logging.getLogger(__name__)


def get_dataset(config: DatasetConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Load custom dataset for training.
    
    Args:
        config: Dataset configuration with training_data_path and validation_data_path
        tokenizer: Tokenizer (typically unused - train_on_responses_only handles tokenization)
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
        Each sample must have a 'text' field with the complete formatted prompt + response
    
    Example implementation for JSON files with {"instruction": ..., "response": ...} format:
    """
    # TODO: Replace this with your actual implementation
    raise NotImplementedError(
        "Please implement get_dataset() in custom_dataset.py\n"
        "See the example below for guidance."
    )
    
    # Example implementation (uncomment and modify):
    """
    # Load training data
    train_path = Path(config.training_data_path)
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    # Load validation data  
    val_path = Path(config.validation_data_path)
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    
    # Format samples for Llama-3 Instruct
    def format_sample(item):
        return {
            "text": (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{item['instruction']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{item['response']}<|eot_id|>"
            )
        }
    
    train_samples = [format_sample(item) for item in train_data]
    val_samples = [format_sample(item) for item in val_data]
    
    logger.info(f"Loaded {len(train_samples)} train, {len(val_samples)} validation samples")
    
    return DatasetDict({
        "train": Dataset.from_list(train_samples),
        "validation": Dataset.from_list(val_samples),
    })
    """