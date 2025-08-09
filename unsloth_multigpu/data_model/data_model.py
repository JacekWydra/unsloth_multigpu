from pydantic import BaseModel
from unsloth import UnslothTrainingArguments


class PretrainedModelConfig(BaseModel):
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_length: int = 4096
    dtype: str | None = None
    load_in_4bit: bool = True
    kwargs: dict | None = None


class PEFTConfig(BaseModel):
    r: int = 16
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    lora_alpha: int = 16
    lora_dropout: int = 0
    bias: str = "none"
    use_gradient_checkpointing: str | bool = "unsloth"
    random_state: int = 3407
    use_rslora: bool = True
    kwargs: dict | None = None


class DatasetConfig(BaseModel):
    use_debug_dataset: bool = True  # Use debug dataset for testing
    unsloth_chat_template: str | None = None
    training_data_path: str = "/path/to/train.json"  # Path to training data
    validation_data_path: str = "/path/to/val.json"  # Path to validation data
    # if train on responses is True
    instruction_part: str | None = None
    response_part: str | None = None


class TrainingConfig(BaseModel):
    trainer_config: UnslothTrainingArguments
    devices: list[int] = [0, 1]
    train_on_responses: bool = False
    peft_config: PEFTConfig = PEFTConfig()
    pretrained_model_config: PretrainedModelConfig = PretrainedModelConfig()
    dataset_config: DatasetConfig

