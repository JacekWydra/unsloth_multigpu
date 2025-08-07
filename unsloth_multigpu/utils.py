from typing import TypeVar, Type
import logging
from pydantic import BaseModel
from unsloth_multigpu.data_model.data_model import TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def load_training_config(config_path: str) -> TrainingConfig:
    return load_config(config_path, TrainingConfig)

# TODO
# def load_evaluation_config(config_path: str) -> EvaluationConfig:
    # return load_config(config_path, EvaluationConfig)

# TODO
def load_config(config_path: str, config_type: Type[T]) -> T:
    ...