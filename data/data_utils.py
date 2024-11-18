import os
import json
import random
import uuid
from itertools import chain
from copy import deepcopy
from functools import lru_cache
import re
import struct

import numpy as np
import torch

from datasets import Features, Sequence, Value
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk, concatenate_datasets

from model import BinBBTTokenizer

# BinBBT data process, v2 model use byte level data, v1 model use patch level

def load_and_prepare_dataset(task_type, file_paths, tokenizer, block_size, patch_size, step_size, streaming=False, cache_dir=None, num_proc=None):
    # Ensure task_type is valid
    assert task_type in ["byte", "text", "classification", "regression"]

    def process_data(data):
        # Extract text from data or raise an error if no text is found
        text = data.get("text") or data.get("data")
        if not text:
            raise ValueError("No text found in data")
        
        # Tokenize the text using the provided tokenizer
        encoded_input = tokenizer(text, add_special_tokens=True, return_attention_mask=False)
        input_ids = encoded_input["input_ids"]

        # Handle label processing for classification and regression tasks
        if task_type == "classification":
            label = data.get("label")
            if not label:
                raise ValueError("No label found in data")
            label = int(label)
        elif task_type == "regression":
            label = data.get("label")
            if not label:
                raise ValueError("No label found in data")
            label = float(label)

        # Return input_ids and label for classification/regression, or None for byte/text tasks
        if task_type in ["classification", "regression"]:
            return input_ids, label
        elif task_type in ["byte", "text"]:
            return input_ids, None

    def group_texts(input_ids_list):
        # Concatenate all input_ids and adjust the total length to fit into blocks
        concatenated_input_ids = list(chain(*input_ids_list))
        total_length = len(concatenated_input_ids)
        if total_length >= block_size * patch_size:
            total_length = (total_length // (block_size * patch_size)) * block_size * patch_size
        # Split the input into chunks of block_size * patch_size
        return [concatenated_input_ids[i: i + block_size * patch_size] for i in range(0, total_length, block_size * patch_size)]

    def create_batches(input_ids_chunks, label=None):
        # Create batches of input data and apply padding for tasks requiring it
        for chunk in input_ids_chunks:
            if task_type in ["byte", "text"]:
                # For byte/text tasks, no padding is required
                yield {"input_ids": chunk, "label_ids": chunk.copy(), "attention_mask": None}
            else:
                # Pad the input data for classification/regression tasks
                batch_encoding = tokenizer.pad({"input_ids": chunk},
                                               padding="max_length",
                                               max_length=block_size * patch_size,
                                               return_attention_mask=True)
                input_ids = batch_encoding["input_ids"]
                attention_mask = batch_encoding["attention_mask"]

                # Adjust attention mask if patch_size is greater than 1
                if patch_size > 1:
                    attention_mask = np.array(attention_mask).reshape(block_size, patch_size)
                    attention_mask = (attention_mask.sum(axis=1) > 0).astype(np.int8)

                yield {"input_ids": input_ids, "label": label, "attention_mask": attention_mask}

    def generate_data(shards):
        # Generate data by reading from file shards
        for shard in shards:
            try:
                # Open each file shard
                with open(shard, 'r', encoding='utf-8') as f:
                    input_ids_list = []
                    label_list = []
                    # Process each line in the file
                    for line in f:
                        try:
                            data = json.loads(line)
                            input_ids, label = process_data(data)
                            input_ids_list.append(input_ids)
                            if task_type in ["classification", "regression"]:
                                label_list.append(label)
                        except (json.JSONDecodeError, ValueError):
                            # Skip invalid data
                            print(f"Warning: Skipping invalid data in {shard}, data {line}")
                    
                    # Group input_ids for byte/text tasks and generate batches
                    if task_type in ["byte", "text"]:
                        input_ids_chunks = group_texts(input_ids_list)
                        yield from create_batches(input_ids_chunks)
                    else:
                        # Generate batches for classification/regression tasks
                        for input_ids, label in zip(input_ids_list, label_list):
                            input_ids_chunks = [input_ids[i:i + block_size * patch_size]
                                                for i in range(0, len(input_ids), step_size)]
                            yield from create_batches(input_ids_chunks, label)
            except FileNotFoundError:
                # Handle file not found error
                print(f"Error: File {shard} not found.")
                continue
            except IOError:
                # Handle file reading errors
                print(f"Error: Could not read file {shard}.")
                continue

    # Define dataset features based on task type
    if task_type in ["byte", "text"]:
        features = Features({
            "input_ids": Sequence(Value(dtype="int16")),
            "label_ids": Sequence(Value(dtype="int16")),
            "attention_mask": Sequence(Value(dtype="int8"))
        })
    elif task_type == "classification":
        features = Features({
            "input_ids": Sequence(Value(dtype="int16")),
            "label": Value(dtype="int32"),
            "attention_mask": Sequence(Value(dtype="int8"))
        })
    elif task_type == "regression":
        features = Features({
            "input_ids": Sequence(Value(dtype="int16")),
            "label": Value(dtype="float32"),
            "attention_mask": Sequence(Value(dtype="int8"))
        })

    # Load the dataset, either streaming or from cached files
    if streaming:
        dataset = IterableDataset.from_generator(generate_data, features=features, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    else:
        dataset = Dataset.from_generator(generate_data, features=features, cache_dir=cache_dir, num_proc=num_proc, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42)

    # Set the dataset format to PyTorch
    dataset = dataset.with_format("torch")
    return dataset


def load_and_prepare_datasets(task_type, train_dir, validation_dir, test_dir,
                              tokenizer, block_size, patch_size, step_size,
                              streaming=False, cache_dir=None, num_proc=None):
    # Helper function to collect all files with '.jsonl' extension from a directory
    def collect_files(directory):
        files = []
        for root, dirs, files_list in os.walk(directory):
            # Iterate over all files in the directory
            for file in files_list:
                # Collect only JSONL files
                if file.endswith('.jsonl'):
                    files.append(os.path.join(root, file))
        return files

    # Collect the list of files for train, validation, and test datasets
    train_files = collect_files(train_dir) if train_dir else []
    validation_files = collect_files(validation_dir) if validation_dir else []
    test_files = collect_files(test_dir) if test_dir else []

    # Raise an error if no files are found in the provided directories
    if not train_files and train_dir:
        raise ValueError("No training files found.")
    if not validation_files and validation_dir:
        raise ValueError("No validation files found.")
    if not test_files and test_dir:
        raise ValueError("No test files found.")

    # Prepare datasets using the provided files for each dataset type
    def prepare_dataset(files):
        return load_and_prepare_dataset(
            task_type,
            files,
            tokenizer,
            block_size,
            patch_size,
            step_size,
            streaming=streaming,
            cache_dir=cache_dir,
            num_proc=num_proc
        )

    # Load and prepare the datasets for training, validation, and testing
    train_dataset = prepare_dataset(train_files) if train_files else None
    validation_dataset = prepare_dataset(validation_files) if validation_files else None
    test_dataset = prepare_dataset(test_files) if test_files else None

    # Return the datasets wrapped in either an IterableDatasetDict or DatasetDict depending on streaming
    if streaming:
        return IterableDatasetDict({
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset
        })
    else:
        return DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset
        })


def get_arrow_dataset(data_dir, shuffle=False):
    # Load a dataset from the disk and optionally shuffle it
    dataset = load_from_disk(data_dir)
    if shuffle:
        # Shuffle the dataset if specified
        dataset = dataset.shuffle(seed=42)
    # Set the dataset format to PyTorch
    dataset = dataset.with_format("torch")
    return dataset


def preprocess_dataset(file_paths, task_type, save_dir,
                       tokenizer, block_size, patch_size, step_size,
                       streaming=False, cache_dir=None, num_proc=None, max_shard_size=None):
    # Ensure that there are file paths provided
    assert len(file_paths) > 0

    # Load and prepare the dataset
    ds = load_and_prepare_dataset(
        task_type=task_type,
        file_paths=file_paths,
        tokenizer=tokenizer,
        block_size=block_size,
        patch_size=patch_size,
        step_size=step_size,
        streaming=streaming,
        cache_dir=cache_dir,
        num_proc=num_proc
    )

    # Save the processed dataset to disk, with optional parallel processing and shard size
    ds.save_to_disk(save_dir, num_proc=num_proc, max_shard_size=max_shard_size)
    return ds


def merge_datasets(dataset_path_list, save_dir=None, num_proc=None, max_shard_size=None, shuffle=False):
    # Initialize a list to collect all datasets
    all_datasets = []
    # Load each dataset from the list of paths
    for dataset_path in dataset_path_list:
        dataset = load_from_disk(dataset_path)
        all_datasets.append(dataset)
    
    # Concatenate all datasets along axis 0 (combine them vertically)
    ds = concatenate_datasets(all_datasets, axis=0)
    # Optionally shuffle the concatenated dataset
    if shuffle:
        ds = ds.shuffle(seed=42)
    # Optionally save the dataset to disk
    if save_dir:
        ds.save_to_disk(save_dir, num_proc=num_proc, max_shard_size=max_shard_size)
    return ds


def merge_dataset_dict(save_dir, train_list, validation_list, test_list, num_proc=None, max_shard_size=None):
    # Merge the datasets for train, validation, and test separately
    train_dataset = merge_datasets(train_list)
    validation_dataset = merge_datasets(validation_list)
    test_dataset = merge_datasets(test_list)
    
    # Combine the datasets into a DatasetDict
    ds = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})
    # Save the DatasetDict to disk
    ds.save_to_disk(save_dir, num_proc=num_proc, max_shard_size=max_shard_size)
    return ds


if __name__=="__main__":
    task_type = "byte"
    train_dir = "/root/shared/ldh_data/datasets/slim/train"
    train_save = "/root/shared/ldh_data/datasets/slim/binbbt_text_arrow_2048_1/train"
    validation_dir = "/root/shared/ldh_data/datasets/slim/validation"
    validation_save = "/root/shared/ldh_data/datasets/slim/binbbt_text_arrow_2048_1/validation"
    test_dir = "/root/shared/ldh_data/datasets/slim/test"
    test_save = "/root/shared/ldh_data/datasets/slim/binbbt_text_arrow_2048_1/test"
    data_dir = "/root/shared/ldh_data/datasets/slim/binbbt_text_arrow_2048_1"
    block_size = 2048
    patch_size = 1
    step_size = 2048 * 1
    streaming = False
    cache_dir = "/root/shared/ldh_data/datasets/slim/cache"
    num_proc = 16
    max_shard_size = "4GB"
    tokenizer = BinBBTTokenizer()

    torch.set_printoptions(threshold=torch.inf)

    train_files = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.jsonl'):
                train_files.append(os.path.join(root, file))
    assert len(train_files) > 0
    validation_files = []
    for root, dirs, files in os.walk(validation_dir):
        for file in files:
            if file.endswith('.jsonl'):
                validation_files.append(os.path.join(root, file))
    assert len(validation_files) > 0
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.jsonl'):
                test_files.append(os.path.join(root, file))
    assert len(test_files) > 0

    # when trainning, train and validation are needed
    preprocess_dataset(
        file_paths=train_files, task_type=task_type, save_dir=train_save,
        tokenizer=tokenizer, block_size=block_size, patch_size=patch_size, step_size=step_size,
        streaming=streaming, cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size
    )
    preprocess_dataset(
        file_paths=validation_files, task_type=task_type, save_dir=validation_save,
        tokenizer=tokenizer, block_size=block_size, patch_size=patch_size, step_size=step_size,
        streaming=streaming, cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size
    )
    preprocess_dataset(
        file_paths=test_files, task_type=task_type, save_dir=test_save,
        tokenizer=tokenizer, block_size=block_size, patch_size=patch_size, step_size=step_size,
        streaming=streaming, cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size
    )

    # data_dir = "/mnt/afs/ldh/data/lattice/binbbt_text_arrow_512_16"
    # data_dir = "/mnt/afs/ldh/data/gyro/binbbt_text_arrow_512_16"
    # data_dict = get_arrow_dataset(data_dir)
    # for data in data_dict["validation"]:
    #     input_ids = data["input_ids"]
    #     print(input_ids.shape)
    #     decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    #     print(decoded)

    # datasets_list = [
    #     "/mnt/afs/ldh/data/gyro/binbbt_text_arrow_512_16/train",
    #     "/mnt/afs/ldh/data/lattice/binbbt_text_arrow_512_16/train",
    #     "/mnt/afs/ldh/data/particle_physics/binbbt_text_arrow_512_16/train",
    #     "/mnt/afs/ldh/data/thepile/llama3_text_arrow_512_16/00",
    # ]
    
    # merge_datasets(datasets_list, save_dir=data_dir, num_proc=num_proc, max_shard_size=max_shard_size, shuffle=True)

# need python path if your python cannot find BinBBTTokenizer
# export PYTHONPATH="/root/shared/ldh_data/binbbt:${PYTHONPATH}"