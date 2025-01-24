import os
import torch
from data.data_utils import preprocess_dataset
if __name__=="__main__":
    task_type = "classification"
    train_dir = "./data/train"
    validation_dir = "./data/validation"
    test_dir = "./data/test"
    train_save_dir = "./data/processed/train"
    validation_save_dir = "./data/processed/validation"
    test_save_dir = "./data/processed/test"
    block_size = 512
    patch_size = 16
    step_size = block_size * patch_size
    pad_id = 0
    streaming = False
    cache_dir = "./data/cache"
    num_proc = 11
    max_shard_size = "2GB"

    
    train_files = []
    for root, dirs, files in os.walk(train_dir):
         for file in files:
             if file.endswith('.jsonl'):
                 train_files.append(os.path.join(root, file))
    assert len(train_files) > 0
    preprocess_dataset(file_paths=train_files, dataset_type=task_type, save_dir=train_save_dir,
                    block_size=block_size, patch_size=patch_size, step_size=step_size, pad_id=0,
                    cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size)
    
    validation_files = []
    for root, dirs, files in os.walk(validation_dir):
        for file in files:
            if file.endswith('.jsonl'):
                validation_files.append(os.path.join(root, file))
    assert len(validation_files) > 0
    preprocess_dataset(file_paths=validation_files, dataset_type=task_type, save_dir=validation_save_dir,
                    block_size=block_size, patch_size=patch_size, step_size=step_size, pad_id=0,
                    cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size)

    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.jsonl'):
                 test_files.append(os.path.join(root, file))
    assert len(test_files) > 0
    preprocess_dataset(file_paths=test_files, dataset_type=task_type, save_dir=test_save_dir,
                     block_size=block_size, patch_size=patch_size, step_size=step_size, pad_id=0,
                     cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size)
    

    datasets = get_arrow_dataset("./data/processed/train")
    print(datasets)