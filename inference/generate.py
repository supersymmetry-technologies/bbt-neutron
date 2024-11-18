import math
import os
import sys
import datetime
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, IterableDataset, DistributedSampler
import datasets
import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    TrainerCallback,
    default_data_collator,
    DataCollatorWithPadding,
    set_seed,
)
import logging
import transformers.utils.logging as hf_logging
import datasets.utils.logging as ds_logging
from transformers.trainer_utils import get_last_checkpoint

from data import load_and_prepare_datasets, get_arrow_dataset
from model import BinBBTForCausalLM, BinBBTForSequenceClassification, BinBBTConfig, BinBBTTokenizer


# Setting up logging
logger = logging.getLogger("binbbt_logger")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={"help": "Override some existing default config settings when a model is trained from scratch."}
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"})
    torch_dtype: Optional[str] = field(default=None, metadata={"help": "Override the default `torch.dtype` and load the model under this dtype."})
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

@dataclass
class DataTrainingArguments:
    task_type: str = field(metadata={"help": "Task type: byte or text or classification or regression"})
    train_dir: Optional[str] = field(default=None, metadata={"help": "The input training data directory."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "The input validation data directory."})
    data_dir: Optional[str] = field(default=None, metadata={"help": "The input arrow data directory."})
    block_size: Optional[int] = field(default=None, metadata={"help": "Optional input sequence length after tokenization."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    # overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    # validation_split_percentage: Optional[int] = field(default=5, metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."})
    # keep_linebreaks: bool = field(default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."})
    patch_size: Optional[int] = field(default=256, metadata={"help": "Size of each byte patch."})
    step_size: Optional[int] = field(default=256, metadata={"help": "Step size to chunk byte patch."})
    number_classes: Optional[int] = field(default=2, metadata={"help": "Number of classes when doing classification."})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    # https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_dir = training_args.logging_dir
    os.makedirs(log_dir, exist_ok=True)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    ds_logging.set_verbosity(log_level)
    hf_logging.set_verbosity(log_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    hf_logging.enable_progress_bar()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"generate.log"))
    file_handler.setLevel(log_level)
    formatter = logging.Formatter(f"[Rank {training_args.local_rank}] [%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    hf_logging.add_handler(console_handler)
    hf_logging.add_handler(file_handler)

    if training_args.local_rank == -1:  # Single process or non-distributed training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:  # Distributed training
        torch.cuda.set_device(training_args.local_rank)
        device = torch.device("cuda", training_args.local_rank)
        # torch.distributed.init_process_group(backend="nccl")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}, "
        + f"16-bit (mixed) precision training: {training_args.bf16}, per_device_train_batch_size: {training_args.per_device_train_batch_size}, "
        + f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps},"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model
    logger.info("start load model")
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "padding_side": "left",
    }
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
    }
    if model_args.model_name_or_path:
        tokenizer = BinBBTTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        config = BinBBTConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        if model_args.tokenizer_name:
            tokenizer = BinBBTTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        else:
            tokenizer = BinBBTTokenizer(**tokenizer_kwargs)
        tokenizer.init_kwargs["auto_map"]  = {"AutoTokenizer": ["tokenization_binbbt.BinBBTTokenizer", "tokenization_binbbt.BinBBTTokenizer"]}
        if model_args.config_name:
            config = BinBBTConfig.from_pretrained(model_args.config_name, **config_kwargs)
        else:
            config = BinBBTConfig(vocab_size = len(tokenizer),
                                    pad_token_id = tokenizer.pad_token_id,
                                    eos_token_id = tokenizer.eos_token_id,
                                    bos_token_id = tokenizer.bos_token_id,
                                    unk_token_id = tokenizer.unk_token_id)
        config.auto_map = {
            "AutoTokenizer": ["tokenization_binbbt.BinBBTTokenizer", "tokenization_binbbt.BinBBTTokenizer"],
            "AutoConfig": "configuration_binbbtv2.BinBBTConfig",
            "AutoModel": "modeling_binbbtv2.BinBBTForCausalLM",
            "AutoModelForCausalLM": "modeling_binbbtv2.BinBBTForCausalLM",
            "AutoModelForSequenceClassification": "modeling_binbbtv2.BinBBTForSequenceClassification"
        }
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.warning(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.warning(f"New config: {config}")

    if data_args.task_type in ["byte", "text"]:
        if model_args.model_name_or_path:
            model = BinBBTForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                      config=config,
                                                      cache_dir=model_args.cache_dir)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from pretrained - Total size={n_params/2**20:.2f}M params")
        else:
            model = BinBBTForCausalLM(config)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    elif data_args.task_type in ["regression", "classification"]:
        config.num_labels = data_args.number_classes
        if data_args.task_type == "regression":
            data_args.number_classes = 1
            config.num_labels = 1
        if model_args.model_name_or_path:
            model = BinBBTForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                                    config=config,
                                                                    cache_dir=model_args.cache_dir,
                                                                    ignore_mismatched_sizes=True)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from pretrained - Total size={n_params/2**20:.2f}M params")
        else:
            model = BinBBTForSequenceClassification(config)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    model.to(device)
    logger.info("end load model")

    embedding_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    logger.info("start load datasets")
    with training_args.main_process_first(desc="dataset map"):
        # datasets = load_and_prepare_datasets(task_type=data_args.task_type,
        #                                      train_dir=data_args.train_dir,
        #                                      validation_dir=data_args.validation_dir,
        #                                      test_dir=None,
        #                                      block_size=data_args.block_size,
        #                                      patch_size=data_args.patch_size,
        #                                      step_size=data_args.step_size,
        #                                      pad_id=0,
        #                                      streaming=data_args.streaming,
        #                                      cache_dir=None,
        #                                      num_proc=data_args.preprocessing_num_workers)
        datasets = get_arrow_dataset(data_args.data_dir)
    logger.info("end load datasets")

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        logger.info("start select eval_dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None and data_args.streaming==False:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(eval_dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())
        else:
            sampler = None
        dataloader = DataLoader(
            eval_dataset, 
            batch_size=training_args.per_device_eval_batch_size, 
            sampler=sampler,
            shuffle=(sampler is None)
        )
        logger.info("end select eval_dataset")

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            if data_args.task_type in ["byte", "text"]:
                return logits.argmax(dim=-1)
            if data_args.task_type == "classification":
                return logits.argmax(dim=-1)
            if data_args.task_type == "regression":
                return logits

        logger.info("start load metric")
        if data_args.task_type in ["byte", "text", "classification"]:
            metric = evaluate.load("../evaluate/metrics/accuracy")
        elif data_args.task_type in ["regression"]:
            metric = evaluate.load("../evaluate/metrics/mse")
        else:
            raise ValueError("task_type must in ['byte', 'text', 'classification', 'regression'].")
        logger.info("end load metric")

        def compute_metrics(eval_preds):
            if data_args.task_type in ["byte", "text"]:
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1) has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)
            if data_args.task_type == "classification":
                preds, labels = eval_preds
                # cm = metric.compute(predictions=preds, references=labels, normalize="true")
                # cm["confusion_matrix"] = cm["confusion_matrix"].tolist()
                return metric.compute(predictions=preds, references=labels)
            if data_args.task_type == "regression":
                preds, labels = eval_preds
                return metric.compute(predictions=preds, references=labels)

    input_len = data_args.block_size * data_args.patch_size
    attention_len = data_args.block_size

    # generate
    for data in dataloader:
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["label_ids"].to(device)

        # print("input_ids shape:", input_ids.shape)
        # print("attention_mask shape:", attention_mask.shape)
        # print("labels shape:", labels.shape)

        truncated_input_ids = input_ids[:, input_len//2:]
        truncated_attention_mask = attention_mask[:, attention_len//2:]
        # outputs = model(input_ids=input_ids, labels=labels)
        # loss = outputs["loss"]
        # logits = outputs["logits"]
        # print(logits.shape)

        output_ids = model.generate(
            truncated_input_ids,
            attention_mask=truncated_attention_mask,
            max_new_tokens=1024,
            num_return_sequences=1,
            do_sample=False,
            top_k=50,
            top_p=0.95
        )
        # print("output_ids shape:", output_ids.shape)
        # print("truncated_input_ids shape:", truncated_input_ids.shape)

        output_list = output_ids.cpu().numpy().tolist()
        input_list = truncated_input_ids.cpu().numpy().tolist()

        for i in range(len(output_list)):
            input_str = tokenizer.decode(input_list[i])
            output_str = tokenizer.decode(output_list[i])
            print("============")
            print(input_str)
            print("------------")
            print(output_str)


if __name__ == "__main__":
    main()