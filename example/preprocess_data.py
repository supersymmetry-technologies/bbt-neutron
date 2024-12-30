import os
import torch
from data.data_utils import preprocess_dataset, BinBBTTokenizer

if __name__ == "__main__":
    # Configuration parameters
    task_type = "classification"
    train_dir = "./data/train"  # Replace with your training data directory
    train_save = "./data/processed/train"  # Replace with your processed training data save directory
    validation_dir = "./data/validation"  # Replace with your validation data directory
    validation_save = "./data/processed/validation"  # Replace with your processed validation data save directory
    test_dir = "./data/test"  # Replace with your test data directory
    test_save = "./data/processed/test"  # Replace with your processed test data save directory
    block_size = 2048
    patch_size = 1
    step_size = 2048 * 1
    streaming = False
    cache_dir = "./data/cache"  # Replace with your cache directory
    num_proc = 4  # Adjust based on your machine configuration
    max_shard_size = "4GB"

    # Initialize Tokenizer
    tokenizer = BinBBTTokenizer()

    # Set PyTorch print options
    torch.set_printoptions(threshold=torch.inf)

    # Load data files
    def load_files(directory):
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.jsonl'):
                    files.append(os.path.join(root, filename))
        return files

    train_files = load_files(train_dir)
    validation_files = load_files(validation_dir)
    test_files = load_files(test_dir)

    # Check if the file lists are empty
    assert len(train_files) > 0, f"No JSONL files found in {train_dir}"
    assert len(validation_files) > 0, f"No JSONL files found in {validation_dir}"
    assert len(test_files) > 0, f"No JSONL files found in {test_dir}"

    # Preprocess and save datasets
    preprocess_dataset(
        file_paths=train_files, task_type=task_type, save_dir=train_save,
        tokenizer=tokenizer, block_size=block_size, patch_size=patch_size, step_size=step_size,
        streaming=streaming, cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size
    )
    print(f"Train dataset processed and saved to {train_save}")

    preprocess_dataset(
        file_paths=validation_files, task_type=task_type, save_dir=validation_save,
        tokenizer=tokenizer, block_size=block_size, patch_size=patch_size, step_size=step_size,
        streaming=streaming, cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size
    )
    print(f"Validation dataset processed and saved to {validation_save}")

    preprocess_dataset(
        file_paths=test_files, task_type=task_type, save_dir=test_save,
        tokenizer=tokenizer, block_size=block_size, patch_size=patch_size, step_size=step_size,
        streaming=streaming, cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size
    )
    print(f"Test dataset processed and saved to {test_save}")
