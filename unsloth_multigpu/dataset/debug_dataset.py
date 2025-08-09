"""Debug dataset for testing the training pipeline."""

import logging
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from unsloth_multigpu.data_model.data_model import DatasetConfig

logger = logging.getLogger(__name__)


def get_dataset(config: DatasetConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    """Create a debug dataset for testing with enough samples for multi-GPU training."""
    # Expanded sample data in Llama-3 Instruct format for multi-GPU testing
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
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is deep learning?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Deep learning is a subset of machine learning that uses neural networks with multiple layers.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the speed of light?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "The speed of light in vacuum is approximately 299,792,458 meters per second.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "How do neural networks work?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Neural networks work by connecting artificial neurons that process and transmit information through weighted connections.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the largest planet in our solar system?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Jupiter is the largest planet in our solar system.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "Explain quantum computing.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the chemical symbol for gold?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "The chemical symbol for gold is Au, from the Latin word 'aurum'.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "How many continents are there?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "There are seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is blockchain?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Blockchain is a distributed ledger technology that maintains a continuously growing list of records in blocks.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is DNA?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "DNA (Deoxyribonucleic acid) is the molecule that carries genetic instructions for life.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is photosynthesis?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the freezing point of water?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "The freezing point of water is 0 degrees Celsius or 32 degrees Fahrenheit.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is renewable energy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Renewable energy comes from sources that are naturally replenished, like solar, wind, and hydroelectric power.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is gravity?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Gravity is the force that attracts objects with mass toward each other, keeping us grounded on Earth.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the periodic table?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "The periodic table is an organized arrangement of all known chemical elements by atomic number.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is cloud computing?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Cloud computing provides on-demand access to computing resources like servers and storage over the internet.<|eot_id|>"
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
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the boiling point of water?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "The boiling point of water is 100 degrees Celsius or 212 degrees Fahrenheit at sea level.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the smallest unit of matter?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "The atom is considered the smallest unit of matter that retains the properties of an element.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is natural language processing?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Natural language processing (NLP) is a branch of AI that helps computers understand and generate human language.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the moon made of?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "The moon is made primarily of rock and metal, with a composition similar to Earth's mantle.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is evolution?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "Evolution is the process by which species change and develop over successive generations through natural selection.<|eot_id|>"
            )
        },
        {
            "text": (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                "What is the internet?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                "The internet is a global network of interconnected computers that allows worldwide communication and information sharing.<|eot_id|>"
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