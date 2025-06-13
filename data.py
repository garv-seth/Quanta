"""
Module for loading and preparing text datasets for the Quasar d-SLM.
This module uses the Hugging Face `datasets` library to handle data loading.
"""

from datasets import load_dataset, Dataset, DatasetDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_text_dataset(dataset_name: str = "roneneldan/TinyStories", split: str = "train", streaming: bool = True):
    """
    Loads a text dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
        split (str): The dataset split to load (e.g., 'train', 'validation').
        streaming (bool): Whether to stream the dataset to avoid downloading it all at once.

    Returns:
        A Hugging Face `Dataset` object.
    """
    try:
        logging.info(f"Attempting to load dataset: {dataset_name} (split: {split}, streaming: {streaming})")
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        logging.info("Successfully loaded dataset.")
        # If streaming, we can't easily check the first element without starting the stream.
        # For non-streaming, you could inspect with: logging.info(f"First example: {next(iter(dataset))}")
        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset '{dataset_name}': {e}")
        logging.info("Returning a small dummy dataset as a fallback.")
        return _create_dummy_dataset()

def _create_dummy_dataset() -> DatasetDict:
    """Creates a small, dummy dataset for debugging and testing."""
    dummy_data = {
        "train": Dataset.from_dict({
            "text": [
                "Once upon a time, in a land of code, a small model learned to write stories.",
                "It read many books and practiced every day.",
                "Its goal was to create something new and wonderful.",
            ]
        }),
        "validation": Dataset.from_dict({
            "text": [
                "This is a validation sentence.",
                "This is another one.",
            ]
        })
    }
    return DatasetDict(dummy_data)

if __name__ == '__main__':
    # Example of how to use the loader
    print("Loading the default TinyStories dataset (streaming):")
    streamed_dataset = load_text_dataset()
    # Since it's streamed, we can't get the length, but we can iterate.
    for i, example in enumerate(streamed_dataset):
        if i >= 3:
            break
        print(f"Example {i+1}: {example['text'][:80]}...")

    print("\nLoading a non-streamed dummy dataset for testing:")
    dummy_dataset_dict = _create_dummy_dataset()
    train_dataset = dummy_dataset_dict['train']
    print(f"Dummy dataset has {len(train_dataset)} examples.")
    print(f"First example: {train_dataset[0]['text']}")