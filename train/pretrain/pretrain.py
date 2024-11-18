import math
import os
import sys
import datetime
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
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
    # Parse the arguments for model, data, and training settings
    # Arguments can be found in src/transformers/training_args.py or by using the --help flag
    # More details: https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_dir = training_args.logging_dir
    os.makedirs(log_dir, exist_ok=True)  # Create the log directory if it doesn't exist
    log_level = training_args.get_process_log_level()  # Set logging level based on process rank
    logger.setLevel(log_level)
    ds_logging.set_verbosity(log_level)
    hf_logging.set_verbosity(log_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    hf_logging.enable_progress_bar()

    # Set up console and file logging handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"pretrain.log"))
    file_handler.setLevel(log_level)
    formatter = logging.Formatter(f"[Rank {training_args.local_rank}] [%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    hf_logging.add_handler(console_handler)
    hf_logging.add_handler(file_handler)

    # Log a summary of the process
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}, "
        + f"16-bit (mixed) precision training: {training_args.bf16}, per_device_train_batch_size: {training_args.per_device_train_batch_size}, "
        + f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps},"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect the last checkpoint to resume training
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

    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # Load pretrained model or tokenizer configuration
    logger.info("start load model")
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "padding_side": "left",
    }
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
    }

    if model_args.model_name_or_path:
        # Load pretrained tokenizer and model configuration from a specified path
        tokenizer = BinBBTTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        config = BinBBTConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        # Initialize a new tokenizer and configuration if no pretrained path is provided
        if model_args.tokenizer_name:
            tokenizer = BinBBTTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        else:
            tokenizer = BinBBTTokenizer(**tokenizer_kwargs)
        tokenizer.init_kwargs["auto_map"] = {"AutoTokenizer": ["tokenization_binbbt.BinBBTTokenizer", "tokenization_binbbt.BinBBTTokenizer"]}
        
        if model_args.config_name:
            config = BinBBTConfig.from_pretrained(model_args.config_name, **config_kwargs)
        else:
            config = BinBBTConfig(
                vocab_size=len(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                unk_token_id=tokenizer.unk_token_id
            )

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

    # Initialize the model based on the task type (causal language modeling or classification/regression)
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
    logger.info("end load model")

    # Adjust token embeddings to match the tokenizer size
    embedding_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Load datasets for training and evaluation
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

    # Training dataset setup
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        logger.info("start select train_dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None and data_args.streaming==False:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info("end select train_dataset")

    # Evaluation dataset setup
    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        logger.info("start select eval_dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None and data_args.streaming==False:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info("end select eval_dataset")

        # Preprocess logits for metrics based on task type
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            if data_args.task_type in ["byte", "text", "classification"]:
                return logits.argmax(dim=-1)
            if data_args.task_type == "regression":
                return logits

        # Load the appropriate metric for evaluation
        logger.info("start load metric")
        if data_args.task_type in ["byte", "text", "classification"]:
            metric = evaluate.load("../../evaluate/metrics/accuracy")
        elif data_args.task_type == "regression":
            metric = evaluate.load("../../evaluate/metrics/mse")
        else:
            raise ValueError("task_type must be in ['byte', 'text', 'classification', 'regression'].")
        logger.info("end load metric")

        # Compute metrics for evaluation
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if data_args.task_type in ["byte", "text"]:
                # preds have the same shape as the labels, after the argmax(-1) has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)
            elif data_args.task_type == "classification":
                return metric.compute(predictions=preds, references=labels)
            elif data_args.task_type == "regression":
                return metric.compute(predictions=preds, references=labels)

    # Custom callback for logging and checkpointing
    class PrinterCallback(TrainerCallback):
        def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            logger.info("init_end")

        def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            json_string = json.dumps(asdict(state), indent=2, sort_keys=True)

        def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.is_local_process_zero:
                logger.info(f"saved when global_step:{state.global_step}, epoch:{state.epoch}, num_input_tokens_seen:{state.num_input_tokens_seen}")

        def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.is_local_process_zero:
                logger.info("train_begin")

        def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.is_local_process_zero:
                logger.info("train_end")

    # Data collator for padding inputs based on task type
    class BinDataCollatorWithPadding:
        def __init__(self, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            padded = self.tokenizer.pad(features, padding="longest", return_attention_mask=True, return_tensors='pt')
            batch = {
                "input_ids": padded["input_ids"],
                "attention_mask": padded["attention_mask"],
            }
            if data_args.task_type in ["byte", "text"]:
                batch["labels"] = batch["input_ids"].clone()
            elif data_args.task_type in ["classification", "regression"]:
                batch["labels"] = features["label"]
            return batch

    data_collator = BinDataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.block_size * data_args.patch_size)

    # Initialize the Trainer object
    logger.info("Initialize our Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        # data_collator=data_collator,  # Optionally use the custom collator
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        callbacks=[PrinterCallback],
    )

    # Training process
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        logger.info("start train")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        logger.info("end train")
        trainer.save_model()

        # Log training metrics
        metrics = train_result.metrics
        if not data_args.streaming:
            max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation process
    if training_args.do_eval:
        logger.info("*** start evaluate ***")
        metrics = trainer.evaluate()
        logger.info("end evaluate")

        if not data_args.streaming:
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        # Calculate perplexity if available
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()