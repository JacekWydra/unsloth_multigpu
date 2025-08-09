from typing import TypeVar, Type
import logging
import yaml
from pathlib import Path
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

def load_config(config_path: str, config_type: Type[T]) -> T:
    """Load configuration from YAML file and validate with Pydantic model.
    
    Args:
        config_path: Path to YAML configuration file
        config_type: Pydantic model class for validation
        
    Returns:
        Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create and validate configuration object
        config = config_type(**config_data)
        logger.info(f"Configuration loaded successfully: {config_type.__name__}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")