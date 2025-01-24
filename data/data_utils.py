import os
import json
import random
import uuid
from copy import deepcopy
from functools import lru_cache
import re
import struct

import numpy as np
import torch

from datasets import Features, Sequence, Value
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk, concatenate_datasets

#from model import BinBBTTokenizer
def pad_and_truncate(example, block_size, patch_size, pad_id):
    input_bytes = example
    input_length = len(input_bytes)

    # Pad input_bytes and attention_mask to the required length
    padded_input = [pad_id] * (block_size * patch_size - input_length) + list(input_bytes)
    attention_mask = [0] * (block_size * patch_size - input_length) + [1] * input_length

    # Ensure the padded input and attention mask are exactly block_size * patch_size in length
    padded_input = padded_input[:block_size * patch_size]
    attention_mask = attention_mask[:block_size * patch_size]

    return {"input_bytes": padded_input, "attention_mask": attention_mask}
def load_and_prepare_dataset(task_type, file_paths, tokenizer, block_size, patch_size, step_size, streaming=False, cache_dir=None, num_proc=None):
    assert task_type in ["byte", "text", "classification", "regression"]

    def process_data(data):
        text = data.get("text") or data.get("data")
        label = data.get("label")
        encoded_input = tokenizer(text)
        input_ids = encoded_input["input_ids"]

        if label is not None:
            if task_type in ["classification"]:
                label = int(label)
            elif task_type in ["regression"]:
                label = float(label)

        def create_batch(input_ids):
            batch_encoding = tokenizer.pad({"input_ids": input_ids}, padding="max_length", max_length=block_size * patch_size, return_attention_mask=True)
            input_ids = batch_encoding["input_ids"]
            attention_mask = batch_encoding["attention_mask"]

            if patch_size > 1:
                attention_mask = np.array(attention_mask).reshape(block_size, patch_size)
                attention_mask = (attention_mask.sum(axis=1) > 0).astype(np.int8)

            if task_type in ["byte", "text"]:
                return {"input_ids": input_ids, "label_ids": input_ids.copy(), "attention_mask": attention_mask}
            elif task_type in ["classification", "regression"]:
                return {"input_ids": input_ids, "label": label, "attention_mask": attention_mask}

        if len(input_ids) <= block_size * patch_size:
            yield create_batch(input_ids)
        else:
            for i in range(0, len(input_ids) - block_size * patch_size + 1, step_size):
                yield create_batch(input_ids[i:i + block_size * patch_size])

    def generate_data(shards):
        for shard in shards:
            try:
                with open(shard, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            yield from process_data(data)
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON line in file {shard}")
            except FileNotFoundError:
                print(f"Error: File {shard} not found.")
            except IOError:
                print(f"Error: Could not read file {shard}.")

    if task_type in ["byte", "text"]:
        features = Features({
            "input_ids": Sequence(Value(dtype="int16")),
            "label_ids": Sequence(Value(dtype="int16")),
            "attention_mask": Sequence(Value(dtype="int8"))
        })
    elif task_type in ["classification"]:
        features = Features({
            "input_ids": Sequence(Value(dtype="int16")),
            "label": Value(dtype="int32"),
            "attention_mask": Sequence(Value(dtype="int8"))
        })
    elif task_type in ["regression"]:
        features = Features({
            "input_ids": Sequence(Value(dtype="int16")),
            "label": Value(dtype="float32"),
            "attention_mask": Sequence(Value(dtype="int8"))
        })

    if streaming:
        dataset = IterableDataset.from_generator(generate_data, features=features, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    else:
        dataset = Dataset.from_generator(generate_byte_data, features=features, cache_dir=cache_dir,
                                         num_proc=num_proc, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42)
    
    # Apply padding and truncation
    dataset = dataset.map(lambda x: pad_and_truncate(x, block_size, patch_size, pad_id))

    def process_and_reshape(example):
        input_ids = example["input_ids"][:block_size * patch_size]
        attention_mask = example["attention_mask"][:block_size * patch_size]
        label_ids = deepcopy(input_ids)

        return {"input_ids": input_ids, "label_ids": label_ids, "attention_mask": attention_mask}

    # Apply the process_and_reshape function
    dataset = dataset.map(process_and_reshape)
    dataset = dataset.with_format("torch")
    # dataset.set_format(type='torch', columns=['input_ids', 'label_ids', 'attention_mask'])

    return dataset


def get_text_dataset(file_paths, block_size, patch_size, step_size, pad_id,
                     streaming=False, cache_dir=None, num_proc=None):
    def process_text_data(data):
        text_bytes = data["text"].encode('utf-8')
        if len(text_bytes) <= block_size * patch_size:
            example = pad_and_truncate({"input_ids": text_bytes}, block_size, patch_size, pad_id)
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            label_ids = input_ids

            # attention_mask = torch.tensor(attention_mask).reshape(block_size, patch_size)
            # patch_attention_mask = (attention_mask.sum(dim=1) > 0).type(torch.long)

            yield {"input_ids": input_ids, "label_ids": label_ids, "attention_mask": attention_mask}
        else:
            for i in range(0, len(text_bytes) - block_size * patch_size + 1, step_size):
                example = pad_and_truncate({"input_ids": text_bytes[i:i + block_size * patch_size]}, block_size, patch_size, pad_id)
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                label_ids = input_ids
                
                # attention_mask = torch.tensor(attention_mask).reshape(block_size, patch_size)
                # patch_attention_mask = (attention_mask.sum(dim=1) > 0).type(torch.long)

                yield {"input_ids": input_ids, "label_ids": label_ids, "attention_mask": attention_mask}

    def generate_text_data(shards):
        for shard in shards:
            with open(shard, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    yield from process_text_data(data)

    features = Features({
        "input_ids": Sequence(Value(dtype="uint8")),
        "label_ids": Sequence(Value(dtype="uint8")),
        "attention_mask": Sequence(Value(dtype="uint8"))
    })
    # Choose between streaming and non-streaming dataset
    if streaming:
        dataset = IterableDataset.from_generator(generate_text_data, features=features, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    else:
        dataset = Dataset.from_generator(generate_text_data, features=features, cache_dir=cache_dir,
                                         num_proc=num_proc, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42)

    dataset = dataset.with_format("torch")
    return dataset


def get_classification_dataset(file_paths, block_size, patch_size, step_size, pad_id,
                               streaming=False, cache_dir=None, num_proc=None):
    def process_classification_data(data):
        text_bytes = data["data"].encode('utf- 8')
        #print(text_bytes)
        # text = data["data"]
        # replaced_parts = replace_numbers(text)

        # # Convert parts to bytes
        # text_bytes = []
        # for part in replaced_parts:
        #     if isinstance(part, str):
        #         text_bytes.extend(part.encode('utf-8'))  # Encode string parts
        #     else:
        #         text_bytes.extend(part)  # Add binary number parts directly
        label = int(data["label"])
        if len(text_bytes) <= block_size * patch_size:
            example = pad_and_truncate(text_bytes, block_size, patch_size, pad_id)
            input_ids = example["input_bytes"]
            attention_mask = example["attention_mask"]
            
            attention_mask = torch.tensor(attention_mask).reshape(block_size, patch_size)
            patch_attention_mask = (attention_mask.sum(dim=1) > 0).type(torch.long)
            #print(f'input_ids-----{input_ids}')
            yield {"input_ids": input_ids, "label": label, "attention_mask": patch_attention_mask}
        else:
            example = pad_and_truncate(text_bytes[:block_size * patch_size], block_size, patch_size, pad_id)
            input_ids = example["input_bytes"]
            attention_mask = example["attention_mask"]
                
            attention_mask = torch.tensor(attention_mask).reshape(block_size, patch_size)
            patch_attention_mask = (attention_mask.sum(dim=1) > 0).type(torch.long)

            yield {"input_ids": input_ids, "label": label, "attention_mask": patch_attention_mask}
    
    def generate_classification_data(shards):
        for shard in shards:
            with open(shard, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # print(data)
                    yield from process_classification_data(data)

    features = Features({
        "input_ids": Sequence(Value(dtype="uint8")),
        "label": Value(dtype="uint8"),
        "attention_mask": Sequence(Value(dtype="uint8"))
    })
    # Choose between streaming and non-streaming dataset
    if streaming:
        dataset = IterableDataset.from_generator(generate_classification_data, features=features, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    else:
        dataset = Dataset.from_generator(generate_classification_data, features=features, cache_dir=cache_dir,
                                         num_proc=num_proc, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42)

    dataset = dataset.with_format("torch")
    return dataset


def get_regression_dataset(file_paths, block_size, patch_size, step_size, pad_id,
                           streaming=False, cache_dir=None, num_proc=None):
    def process_regression_data(data):
        text_bytes = data["data"].encode('utf-8')
        label = float(data["label"])
        if len(text_bytes) <= block_size * patch_size:
            example = pad_and_truncate({"input_ids": text_bytes}, block_size, patch_size, pad_id)
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]

            attention_mask = torch.tensor(attention_mask).reshape(block_size, patch_size)
            patch_attention_mask = (attention_mask.sum(dim=1) > 0).type(torch.long)

            yield {"input_ids": input_ids, "label": label, "attention_mask": patch_attention_mask}
        else:
            for i in range(0, len(text_bytes) - block_size * patch_size + 1, step_size):
                example = pad_and_truncate({"input_ids": text_bytes[i:i + block_size * patch_size]}, block_size, patch_size, pad_id)
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                
                attention_mask = torch.tensor(attention_mask).reshape(block_size, patch_size)
                patch_attention_mask = (attention_mask.sum(dim=1) > 0).type(torch.long)

                yield {"input_ids": input_ids, "label": label, "attention_mask": patch_attention_mask}
    
    def generate_regression_data(shards):
        for shard in shards:
            with open(shard, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    yield from process_regression_data(data)

    features = Features({
        "input_ids": Sequence(Value(dtype="uint8")),
        "label": Value(dtype="float32"),
        "attention_mask": Sequence(Value(dtype="uint8"))
    })
    # Choose between streaming and non-streaming dataset
    if streaming:
        dataset = IterableDataset.from_generator(generate_regression_data, features=features, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    else:
        dataset = Dataset.from_generator(generate_regression_data, features=features, cache_dir=cache_dir,
                                         num_proc=num_proc, gen_kwargs={"shards": file_paths})
        dataset = dataset.shuffle(seed=42)

    dataset = dataset.with_format("torch")
    return dataset


def load_and_prepare_datasets(task_type, train_dir, validation_dir, test_dir,
                              tokenizer, block_size, patch_size, step_size,
                              streaming=False, cache_dir=None, num_proc=None):
    def collect_files(directory):
        files = []
        for root, dirs, files_list in os.walk(directory):
            for file in files_list:
                if file.endswith('.jsonl'):
                    files.append(os.path.join(root, file))
        return files

    train_files = collect_files(train_dir) if train_dir else []
    validation_files = collect_files(validation_dir) if validation_dir else []
    test_files = collect_files(test_dir) if test_dir else []

    if not train_files and train_dir:
        raise ValueError("No training files found.")
    if not validation_files and validation_dir:
        raise ValueError("No validation files found.")
    if not test_files and test_dir:
        raise ValueError("No test files found.")

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

    train_dataset = prepare_dataset(train_files) if train_files else None
    validation_dataset = prepare_dataset(validation_files) if validation_files else None
    test_dataset = prepare_dataset(test_files) if test_files else None

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


def get_arrow_dataset(data_dir, shuffle=False,dataseed=42):
    dataset = load_from_disk(data_dir)
    if shuffle:
        dataset = dataset.shuffle(seed=dataseed)
    dataset = dataset.with_format("torch")
    return dataset


def preprocess_dataset(file_paths, dataset_type, save_dir,
                       block_size, patch_size, step_size, pad_id=0,
                       cache_dir=None, num_proc=None, max_shard_size=None):
    assert len(file_paths) > 0

    if dataset_type == "byte":
        ds = get_byte_dataset(file_paths, block_size, patch_size, step_size, pad_id,
                              streaming=False, cache_dir=cache_dir, num_proc=num_proc)
    elif dataset_type == "text":
        ds = get_text_dataset(file_paths, block_size, patch_size, step_size, pad_id,
                              streaming=False, cache_dir=cache_dir, num_proc=num_proc)
    elif dataset_type == "classification":
        ds = get_classification_dataset(file_paths, block_size, patch_size, step_size, pad_id,
                                        streaming=False, cache_dir=cache_dir, num_proc=num_proc)
    elif dataset_type == "regression":
        ds = get_regression_dataset(file_paths, block_size, patch_size, step_size, pad_id,
                                    streaming=False, cache_dir=cache_dir, num_proc=num_proc)
    else:
        raise "task_type must in [text, classification, regression, byte]"
    
    ds.save_to_disk(save_dir, num_proc=num_proc, max_shard_size=max_shard_size)
    return ds


def merge_datasets(dataset_path_list, save_dir=None, num_proc=None, max_shard_size=None):
    all_datasets = []
    for dataset_path in dataset_path_list:
        dataset = load_from_disk(dataset_path)
        all_datasets.append(dataset)
    
    ds = concatenate_datasets(all_datasets, axis=0)
    if save_dir:
        ds.save_to_disk(save_dir, num_proc=num_proc, max_shard_size=max_shard_size)
    return ds


def merge_dataset_dict(save_dir, train_list, validation_list, test_list, num_proc=None, max_shard_size=None):
    train_dataset = merge_datasets(train_list)
    validation_dataset = merge_datasets(validation_list)
    test_dataset = merge_datasets(test_list)
    
    ds = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})
    ds.save_to_disk(save_dir, num_proc=num_proc, max_shard_size=max_shard_size)
    return ds


