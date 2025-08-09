"""Dataset selection logic for training."""

import logging
from datasets import DatasetDict
from transformers import AutoTokenizer
from unsloth_multigpu.data_model.data_model import DatasetConfig
from unsloth_multigpu.dataset import debug_dataset, custom_dataset

logger = logging.getLogger(__name__)


def get_training_dataset(config: DatasetConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Select and load the appropriate dataset based on configuration.
    
    Expected format for Llama-3 Instruct:
    ```
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    
    {{ user_message }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    {{ assistant_response }}<|eot_id|>
    ```
    
    Args:
        config: Dataset configuration with use_debug_dataset flag
        tokenizer: Tokenizer (passed to dataset loaders but typically unused)
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    if config.use_debug_dataset:
        logger.info("Loading debug dataset for testing")
        return debug_dataset.get_dataset(config, tokenizer)
    else:
        logger.info("Loading custom dataset")
        try:
            return custom_dataset.get_dataset(config, tokenizer)
        except (ImportError, NotImplementedError) as e:
            logger.error(
                f"Failed to load custom dataset: {e}\n"
                "Please implement get_dataset() in custom_dataset.py "
                "or set use_debug_dataset=true in your config."
            )
            raise