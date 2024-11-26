# BBT-Neutron Simple Guide

A simple instruction and user guide for script running and parameter configuration

## Author：Xinglong Jia

## File structure

```text
bbt-neutron/
│
├── config/
    ├── ds_config_zero1.json
    ├── ds_config_zero2.json
    └── ds_config_zero3.json
├── data/
    ├── train_data/
        ├── example_holdout_0.jsonl
        └── example_holdout_1.jsonl
    ├── validation_data/
        ├── example_holdout_2.jsonl
        └── example_holdout_3.jsonl
    ├── __init__.py
    ├── data_utils.py
    └── download.py
├── inference/
    ├── test.py
    └── test.sh
├── model/
    ├── __init__.py
    ├── configuration_binbbt.py
    └── modelling_binbbt.sh
└── train/
    ├── merge_peft_model
        ├── merge_muilt_peft_adapter.py
        ├── merge_muilt.sh
        ├── merge_peft_adapter.py
        └── merge.sh
    ├── pretrain
        ├── .deepspeed_env
        ├── accuracy.py
        ├── ds_config_zero1.json
        ├── ds_config_zero2.json
        ├── ds_config_zero3.json
        ├── hostfile
        ├── pretrain.py
        └── pretrain.sh
    ├── sft
        ├── accuracy
        ├── ds_config_zero2.json
        ├── finetune_clm_lora.py
        ├── finetune_clm.py
        ├── finetune_lora.sh
        └── finetune.sh
    ├── finetune_lora.py
    └── finetune_sft.py
```

## Data Pre-processing

Binary pre-processing of the `jsonl` format datasets is required, if you use non-streaming training mode (which means reading all the datasets at once).

Data pre-processing is done by directly running `data/data_utils.py`. It's implemented by three functions, namely `preprocess_dataset()`, `merge_datasets()` and `merge_dataset_dict()`.

You need to execute the relevant code under the main() function.

### Parameter Settings

Properly setting the following parameters is often useful. Explanation of each parameter's usage is listed in the following section.

```python
if __name__=="__main__":
    task_type = "text"
    train_dir = "/root/data/datasets/SlimPajama/train/chunk1"
    validation_dir = "/root/data/datasets/SlimPajama/validation"
    test_dir = "/root/data/datasets/SlimPajama/test"
    data_dir = "/root/data/datasets/SlimPajama/train_arrow/chunk1"
    block_size = 4096
    patch_size = 1
    step_size = 4096
    pad_id = 0
    cache_dir = "/root/data/datasets/SlimPajama/cache"
    num_proc = 82
    max_shard_size = "4GB"
```

1. `task_type`: Assign the type of your task. Need to be consistent with the value in `pretrain.sh`.
2. `train_dir`: Assign the path of your training dataset. Files in `jsonl` format under the assigned directory will be used for train_dir.
3. `validation_dir`: Assign the path of your validation dataset. Similar to `train_dir`.
4. `test_dir`: Assign the path of your testing dataset. Similar to `train_dir`.
5. `data_dir`: Assign the path of your output datasets after pre-processing.
6. `block_size`: Set `block_size`. Need to be consistent with the value in `pretrain.sh`.
7. `patch_size`: Set `patch_size`. Need to be consistent with the value in `pretrain.sh`.
8. `step_size`: Set `step_size`. Need to be consistent with the value in `pretrain.sh`.
9. `pad_id`: Set pad token id. Need to be consistent with the value in `pretrain.sh`.
10. `cache_dir`：Assign the path to store your cached datasets during data pre-processing. Remember to reserve enough disk quote for this directory.
11. `num_proc`：Assign the maxium number of processes in the data pre-processing execution.
12. `max_shard_size`：Assign the maxium file size of the output arrow file after data pre-processing.

### Binary Data Pre-processing

To speed up this procedure, you need to manually split a larger `.jsonl` file into several smaller `.jsonl` files, for the convenience of parallel processing.

Input parameters of `preprocess_dataset` function：
1. `file_paths`：A list of paths containing all the `.jsonl` files used for the task.
2. `dataset_type`：Type of the task.
3. `save_dir`：Directory to save the output files after pre-processing.
4. `block_size`：Set block_size, consistent with `pretrain.sh`.
5. `patch_size`：Set patch_size，consistent with `pretrain.sh`.
6. `step_size`：Set step_size，consistent with `pretrain.sh`.
7. `pad_id`：Set pad token id.
8. `cache_dir`：Assign the path to store your cached datasets during data pre-processing.
9. `num_proc`：Assign the maxium number of processes.
10. `max_shard_size`：Assign the maxium file size of the output arrow file after data pre-processing.

A simple processing logic is shown below. You can refer to this logic to write code for your own task. The code will scan all the files in `.jsonl` format under your `train_dir`, and pre-process all the files found.

```python
train_files = []
for root, dirs, files in os.walk(train_dir):
    for file in files:
        if file.endswith('.jsonl'):
            train_files.append(os.path.join(root, file))
assert len(train_files) > 0
preprocess_dataset(file_paths=train_files, dataset_type=task_type, save_dir=data_dir,
                   block_size=block_size, patch_size=patch_size, step_size=step_size, pad_id=0,
                   cache_dir=cache_dir, num_proc=num_proc, max_shard_size=max_shard_size)
```

After data pre-processing, we need to merge the splitted output files.

We can use `merge_datasets` or `merge_dataset_dict` function to merge the output files.
If all the output files are required to be merged into a single dataset, please use `merge_datasets` funtion. Input parameters of this function includes:

1. `dataset_path_list`: Path to save all the output file in arrow format after the binary pre-processing.
2. `save_dir=None`：Path to save the output dataset after the merging process. Set to None if subsequent process is needed.
3. `num_proc=None`：Maxium number of processes.

Now suppose `preprocess_dataset` is executed three times，and the three corresponding `save_dir` are `["train_arrow", "validation_arrows", "test_arrows"]`:

```python
dataset_path_list = ["train__arrows", "validation_arrows", "test_arrows"]
merge_datasets(dataset_path_list, save_dir="dir/to/save/", num_proc=80)
```

If we need to save the output dataset in the `DatasetDict` format，please use `merge_dataset_dict` fucntioin. Input parameters of this function includes:

1. `save_dir`: Path to save the output `DatasetDict` after the merging process.
2. `train_list, validation_list, test_list`: Lists of directory for three different datasets respectively.
3. `num_proc=None`: Maxium number of processes.

Now suppose `preprocess_dataset` is executed for many times，and the output are saved into three groups(with each group containing more than one `save_dir`) `["train_arrow_list", "validation_arrow_list", "test_arrow_list"]`:

```python
all_arrow_list = ["train_arrow_list", "validation_arrow_list", "test_arrow_list"]
merge_dataset_dict(save_dir="dir/to/save", all_arrow_list[0], all_arrow_list[1], all_arrow_list[2], num_proc=80)
```



## Pretrain

Execute the following command to start BBT-Neutron pretraining task in backend mode. `path/to/log` is the path to the log file which records the standard output on the screen.

```bash
nohup bash pretrain.sh > path/to/log 2>&1 &
```

<div style="border: 1px solid #2196F3; padding: 10px; background-color: #e7f3fe; color: #2196F3;">
  <strong>Command Explanation:</strong> 
  
  1. `nohup`: Stand for “no hang up”, enable to continue the command execution after the terminal is closed.
  2. `bash pretrain.sh`: Use bash to execute the `pretrain.sh` script.
  3. `> path/to/log`: Redirect the standard outpout on the screen to an assigned log file.
  4. `2>&1`: Redirect the standard error info to the standard output, which means the standard error info will also be recorded in the log file.
  5. `&`: Enable the program to run in backend mode. In this mode, we can continue to run other commands in the terminal, without waiting for the program to finish first.
</div>

### Configuration Items and Corresponding Description in `pretrain.sh`

#### Output Directory Creation and Files Copy

```bash
output_model=/root/data/local-nvme2/output
```
1. `output_model=/root/data/local-nvme2/output`: Assign the path to save the training output.
```bash
if [ ! -d "${output_model}" ]; then
    mkdir -p "${output_model}"
fi
cp ./pretrain.sh "${output_model}"
cp ./ds_config_zero*.json "${output_model}"
```
2. The script checks if `output_model` exists, and creates the directory if it doesn't exist.
3. Then script then copies `pretrain.sh` and `ds_config_zero*.json` configuration file into `output_model` directory.

#### Environment Parameters Setting

```bash
export PYTHONPATH="/root/bbt-neutron:${PYTHONPATH}"
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export CUDA_HOME=/usr/local/cuda/
export CUTLASS_PATH="/opt/cutlass"
export TORCH_CUDA_ARCH_LIST="8.9 9.0 9.0a"
export TORCH_USE_CUDA_DSA=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3,ib4,ib5,ib6,ib7,ib8
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
export NCCL_IB_GID_INDEX=3
```

1. `PYTHONPATH`：Assign the root path of `bbt-neutron` directory.
2. `CUDA_HOME`：Set CUDA installation path.
3. `CUTLASS_PATH`：Set CUTLASS path (unknown effects).
4. `TORCH_CUDA_ARCH_LIST`：Set CUDA architecture list supported by PyTorch (unknown effects).
5. `TORCH_USE_CUDA_DSA`：Whether to use CUDA DSA (unknown effects).
6. `NCCL_DEBUG`：Set NCCL debug level (WARN by default, unknown effects).
7. `NCCL_SOCKET_IFNAME` 和 `NCCL_IB_HCA`：Set NCCL socket interface and IB HCA (unknown effects).
8. `NCCL_IB_GID_INDEX`：Set IB GID index (unknown effects).

#### DeepSpeed Training Commands and Configuration Items

1. `deepspeed --hostfile ./hostfile`: Assign the file which contains all the hosts information.
    1. If the task is running on a single host:
        1. `--include localhost:0`: Assign the GPU device number(s) to be used in the task.
        2. `--master_port`: Assign the port number of the master node.
    2. If the task is running on multiple hosts:
        1. `--hostfile ./hostfile`: Assign the file which contains the ip address list and available resource information of all the hosts.
        2. `--master_port`: Assign the port number of the master node.
        3. `--include 10.200.1.18:0,1@10.200.1.20:6,7`: Assgin nodes' ip addresses and the GPU device number(s) to be used in the task.
        4. `--ssh_port 22242`: Assign SSH port number.
2. `pretrain.py`: Assign training script.
3. `--task_type classification`: Task type. `classification` for classification task, `regression` for regression task and `text` for language task.
4. `--number_classes 2`: For `classification` task, it represents the number of classes to be classified. For `regression` task, it's manually set to 1.
5. `--train_dir /path/to/train_dir`: Assign the path to the training datasets. Files end with `.jsonl` are used. Each single line of the file is an individual sample.
6. `--validation_dir /path/to/validation_dir`: Assign the path to the validation datasets. Similar to `train_dir`.
7. `--data_dir /data_dir`: Assign the data directory (unknown effects).
8. `--streaming true`: Whether to use streaming data transmission.
9. `--per_device_train_batch_size 64`: Training batch size for each device.
10. `--per_device_eval_batch_size 64`: Validation batch size for each device.
11. `--do_train`: Whether to do training procedure (true by default).
12. `--do_eval`: Whether to do evaluation procedure (true by default).
13. `--output_dir "${output_model}"`: Output directory containing training results.
14. `--use_fast_tokenizer false`: Whether to use fast token (unknown effects).
15. `--max_steps 10000`: Assign total number of training steps here, if streaming data transmission is used.
16. `--max_eval_samples 4600`: Maximum number of evaluation samples.
17. `--learning_rate 1e-4`: Learning rate.
18. `--gradient_accumulation_steps x`: Gradient accumulation steps, which means updating model's parameters once after x batches of samples are processed.
19. `--num_train_epochs x`: Number of training epochs. Doesn't work in streaming data transmission mode.
20. `--warmup_steps 5000`: Number of warmup steps for the learning rate.
21. `--preprocessing_num_workers 128`: Number of working processes in pre-processing stage. Here pre-processing means transferring jsonl format datasets into byte format datasets.
22. `--logging_dir "${output_model}/logs"`: Path to save the tensorboard infomation and training logs.
23. `--logging_strategy "steps"`: Logging strategy.
24. `--logging_steps 100`: If `logging_strategy` is set to `steps`, this parameter assigns how many steps to print a single log information.
25. `--log_level "info"`: Assign log information level.
26. `--save_strategy "steps"`: Checkpoint save strategy. Set to "steps" in the streaming data transmission. 
27. `--save_steps x`: If `save_strategy` is set to `steps`, this parameter assigns how many steps to save a checkpoint.
28. `--save_total_limit 10`: Maximum number of checkpoints can be saved in a single task.
29. `--eval_strategy "steps"`: Evaluation strategy, set to "steps" in the streaming data transmission.
30. `--eval_steps x`: If `eval_strategy` is set to `steps`, this parameter assigns how many steps to do an evaluation procedure.
31. `--seed 42`: Set seed of the random number generator.
32. `--disable_tqdm false`: Where to disable tqdm. Enable tqdm when it's set to `false`.
33. `--ddp_find_unused_parameters false`: DDP find unused parameters (unknown effects).
34. `--block_size 512`: Assign block size, similar to context length in language model.
35. `--patch_size 32`: Assign the number of bytes in each block.
36. `--step_size x`: If the length of a single sample is greater than `block_size * patch_size`, then a new sample is generated every step_size from the beginning to the end of the original sample.
37. `--overwrite_output_dir true`: Where to overwrite the information in the output directory.
38. `--run_name "${output_model}"`: Run name (unknown effects).
39. `--bf16`: Use bf16 format to represent float numbers.
40. `--bf16_full_eval`: Full evaluation when using BF16 format (unknown effects).
41. `--gradient_checkpointing`: Gradient checkpointing (unknown effects).
42. `--deepspeed ./ds_config_zero1.json`: Assign deepspeed configuration file.
43. `--ignore_data_skip`: ignore data skip (unknown effects).
44. `--ddp_timeout 18000000`: Set DDP timeout.
45. `--trust_remote_code`: Trust remote code (unknown effects).
46. `--resume_from_checkpoint ${output_model}/checkpoint-18000`: Resume training from an existing checkpoint.
47. `| tee -a "${output_model}/train.log"`: Add HuggingFace log information to the log file.
