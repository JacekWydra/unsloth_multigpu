"""Debug dataset for testing the training pipeline."""

import logging
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from unsloth_multigpu.data_model.data_model import DatasetConfig

logger = logging.getLogger(__name__)


def get_dataset(config: DatasetConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """Create a minimal debug dataset for testing."""
    # Sample data in Llama-3 Instruct format
    train_samples = [
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "2 + 2 equals 4.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "I'm doing well, thank you for asking! How can I help you today?<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "The capital of France is Paris.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "Explain machine learning.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Machine learning is a type of artificial intelligence where computers learn patterns from data.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is Python?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Python is a high-level programming language known for its simplicity and readability.<|eot_id|>"
            )
        },
    ]
    
    val_samples = [
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is artificial intelligence?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Artificial intelligence is the simulation of human intelligence in machines.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "How many days are in a week?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "There are 7 days in a week.<|eot_id|>"
            )
        },
    ]
    
    dataset = DatasetDict({
        "train": Dataset.from_list(train_samples),
        "validation": Dataset.from_list(val_samples),
    })
    
    logger.info(
        f"Debug dataset loaded - Training: {len(train_samples)} samples, "
        f"Validation: {len(val_samples)} samples"
    )
    
    return dataset