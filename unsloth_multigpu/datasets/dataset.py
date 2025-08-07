from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from trl import apply_chat_template
from unsloth_multigpu.data_model.data_model import DatasetConfig

def get_training_dataset(config: DatasetConfig, tokenizer: AutoTokenizer) -> Dataset:
    """
    This function has to return a DatasetDict. In the example below train_samples and val_samples
    are lists of dictionaries with only one field "text" containing the whole prompt with response
    in a desired format.  
    
    E.g. for llama instruct:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {{ model_answer_1 }}<|eot_id|>

    
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_samples),
            "validation": Dataset.from_list(val_samples),
        }
    )

    Skip tokenization - let train_on_responses_only handle it
    The dataset only needs the "text" field for train_on_responses_only to work

    print(
        f"Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}"
    )
    return dataset
    """
    ...
