# BBT-Neutron 简易指南

针对运行及配置脚本的简易说明和使用指南

## 作者：贾兴隆

## 文件结构

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

## 数据预处理

如果采用非流式数据传输方式（即一次性读入所有数据），则需要进行 `jsonl` 数据的二进制预处理。

通过直接调用 `data/data_utils.py` 来实现数据的预处理，主要由文件内的 `preprocess_dataset()`、`merge_datasets()`、`merge_dataset_dict` 三个函数来实现数据预处理。

在 `if __name__=="__main__":` 下执行相关代码。

### 变量设定

设定下面这些变量通常是有用的，下面的小节会解释它们的用途。

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

1. `task_type`：指定任务类型，需要和 `pretrain.sh` 内的设置保持一致。
2. `train_dir`：指定训练集路径，该路径下存在需要处理的 `jsonl` 文件。
3. `validation_dir`：指定验证集路径，同上。
4. `test_dir`：指定测试集路径，同上。
5. `data_dir`：指定处理后数据存放的路径。
6. `block_size`：设定 `block_size`，需要和 `pretrain.sh` 内的设置保持一致。
7. `patch_size`：设定 `patch_size`，需要和 `pretrain.sh` 内的设置保持一致。
8. `step_size`：设定 `step_size`，需要和 `pretrain.sh` 内的设置保持一致。
9. `pad_id`：设定填充 id，需要和 `pretrain.sh` 内的设置保持一致。
10. `cache_dir`：指定数据预处理时，缓存文件存放的位置，注意要指定一个空间足够大的位置。
11. `num_proc`：指定执行时的最大进程数。
12. `max_shard_size`：数据预处理后，每个 `arrow` 文件的最大大小。

### 二进制化预处理

要想加快处理速度，需要尽量将较大的 `.jsonl` 文件手动分割为多个较小的 `.jsonl` 文件，以便并行运行。

`preprocess_dataset` 函数接收：
1. `file_paths`：所有 `.jsonl` 文件的路径列表；
2. `dataset_type`：任务类型；
3. `save_dir`：预处理后输出文件的存放位置；
4. `block_size`：block_size，同 `pretrain.sh`；
5. `patch_size`：patch_size，同 `pretrain.sh`；
6. `step_size`：step_size，同 `pretrain.sh`；
7. `pad_id`：填充的 pad_id；
8. `cache_dir`：缓存目录；
9. `num_proc`：最大进程数；
10. `max_shard_size`：数据预处理后，每个 `arrow` 文件的最大大小。

下面是一个简单的处理逻辑，可以仿照这个逻辑来编写代码，代码内的变量采用了上方“变量设定”小节中的变量。这些代码将检索 `train_dir` 内所有 `.jsonl` 后缀的文件进行二进制预处理。

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

预处理结束后，还需要对被分割成多个输出的文件进行合并。

使用 `merge_datasets` 或 `merge_dataset_dict` 进行合并。

当只需统一合并为一个数据集时，执行 `merge_datasets` 函数，该函数接收

1. `dataset_path_list`：所有完成二进制化预处理的数据文件（arrow）所存放的位置；
2. `save_dir=None`：合并后数据存放的位置，如果需要进一步处理可设为 None；
3. `num_proc=None`：最大进程数；

继承本节之前出现的所有变量名，并假设 `preprocess_dataset` 被执行三次，保存到了三个 `save_dir` 位置 `["train_arrow", "validation_arrows", "test_arrows"]`：

```python
dataset_path_list = ["train__arrows", "validation_arrows", "test_arrows"]
merge_datasets(dataset_path_list, save_dir="dir/to/save/", num_proc=80)
```

当需要将数据存为 `DatasetDict` 的时候，调用 `merge_dataset_dict` 函数，该函数接收

1. `save_dir`：完成合并后 `DatasetDict` 保存的位置；
2. `train_list, validation_list, test_list`：分别为三种数据集的目录列表；
3. `num_proc=None`：最大进程数；

继承本节之前出现的所有变量名，并假设 `preprocess_dataset` 被执行多次，保存到了三组位置（每一组包含复数个 `save_dir`） `["train_arrow_list", "validation_arrow_list", "test_arrow_list"]`：

```python
all_arrow_list = ["train_arrow_list", "validation_arrow_list", "test_arrow_list"]
merge_dataset_dict(save_dir="dir/to/save", all_arrow_list[0], all_arrow_list[1], all_arrow_list[2], num_proc=80)
```



## 预训练 pretrain

下面的命令可以开始后台运行 BBT-Neutron 的预训练，`path/to/log` 为记录屏幕打印输出的日志文件

```bash
nohup bash pretrain.sh > path/to/log 2>&1 &
```

<div style="border: 1px solid #2196F3; padding: 10px; background-color: #e7f3fe; color: #2196F3;">
  <strong>命令详解:</strong> 
  
  1. `nohup` 意为 “no hang up”，允许关闭终端后继续运行指令。
  2. `bash pretrain.sh`：这是要运行的脚本文件 `pretrain.sh`，使用 bash 来执行。
  3. `> path/to/log`：这个部分将标准输出（stdout）重定向到指定的日志文件 path/to/log。
  4. `2>&1`：这个部分将标准错误输出（stderr）重定向到标准输出（stdout），也就是说，错误信息也会被写入到日志文件中。
  5. `&`：这个符号表示在后台运行命令。这样可以继续在终端中执行其他命令，而不需要等待这个脚本完成。
</div>

### `pretrain.sh` 中的配置项和说明

#### 输出模型目录的创建和文件复制

```bash
output_model=/root/data/local-nvme2/output
```
1. `output_model=/root/data/local-nvme2/output`：指定训练输出和结果的存放位置。
```bash
if [ ! -d "${output_model}" ]; then
    mkdir -p "${output_model}"
fi
cp ./pretrain.sh "${output_model}"
cp ./ds_config_zero*.json "${output_model}"
```
2. 随后脚本将检查 `output_model` 目录是否存在，如果不存在则创建该目录。
3. 接着复制 `pretrain.sh` 脚本和 `ds_config_zero*.json` 配置文件到 `output_model` 目录

#### 环境变量设置

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

1. `PYTHONPATH`：指定 `bbt-neutron` 文件夹的根目录。
2. `CUDA_HOME`：设置 CUDA 的安装路径。
3. `CUTLASS_PATH`：设置 CUTLASS 的路径（未知作用）。
4. `TORCH_CUDA_ARCH_LIST`：设置 PyTorch 支持的 CUDA 架构列表（未知作用）。
5. `TORCH_USE_CUDA_DSA`：启用 CUDA DSA（未知作用）。
6. `NCCL_DEBUG`：设置 NCCL 的调试级别为 WARN（未知作用）。
7. `NCCL_SOCKET_IFNAME` 和 `NCCL_IB_HCA`：配置 NCCL 的网络接口和 HCA（未知作用）。
8. `NCCL_IB_GID_INDEX`：设置 GID 索引（未知作用）。

#### DeepSpeed 训练命令和配置项

1. `deepspeed --hostfile ./hostfile`：指定包含主机信息的文件：
    1. 如果是在单服务器上运行：
        1. `--include localhost:0`：指定需要使用的 GPU 编号。
        2. `--master_port`：设置主节点端口。
    2. 如果是在多服务器上并行运行（暂未测试）：
        1. `--hostfile ./hostfile`：指定服务器地址列表和可用资源信息所在的文件。
        2. `--master_port`：设置主节点端口。
        3. `--include 10.200.1.18:0,1@10.200.1.20:6,7`：指定训练使用的节点和 GPU 编号。
        4. `--ssh_port 22242`：指定 SSH 端口
2. `pretrain.py`：指定训练脚本。
3. `--task_type classification`：任务类型，如果是分类任务此项为 。`classification`，如果是语言任务则为 `text`。
4. `--number_classes 2`：如果选定为分类任务，该项用于指定类别的数量。
5. `--train_dir /path/to/train_dir`：指定训练数据目录，训练样本位于文件夹内的 `jsonl` 文件中，每一行为一个样本。
6. `--validation_dir /path/to/validation_dir`：指定验证（或评估）数据目录，文件夹内容同训练集。
7. `--data_dir /data_dir`：数据目录（未知作用）。
8. `--streaming true`：是否启用流式数据传输。
9. `--per_device_train_batch_size 64`：每个设备的训练批次大小。
10. `--per_device_eval_batch_size 64`：每个设备的评估批次大小。
11. `--do_train`：是否执行训练，空代表 true。
12. `--do_eval`：是否执行评估，空代表 true。
13. `--output_dir "${output_model}"`：训练结果的输出目录，在最开始的 `output_model` 指定即可，也可在此手动修改（不推荐）。
14. `--use_fast_tokenizer false`：不使用快速分词器（未知作用）。
15. `--max_steps 10000`：如果使用流式数据传输，在此指定总训练步数，一步为一个样本（如果序列长度超过 `block_size * patch_size` 则一个原始样本会被切割为多个样本）。如果该数值大于总样本数，则完成所有样本训练后会从头重新开始（猜测）。
16. `--max_eval_samples 4600`：最大验证样本数（未知作用）。
17. `--learning_rate 1e-4`：训练的学习率。
18. `--gradient_accumulation_steps x`：梯度累积步骤数，其含义为累计 x 个批次的梯度，再执行一次更新。
19. `--num_train_epochs x`：训练轮数，流式数据传输下，该项不生效。
20. `--warmup_steps 5000`：预热步骤数（未知作用）。
21. `--preprocessing_num_workers 128`：预处理的工作线程数，这个预处理指的是将 jsonl 中的数据转换成字节数据。
22. `--logging_dir "${output_model}/logs"`：tensorboard 和训练日志（`run.log`）的存放目录。
23. `--logging_strategy "steps"`：训练日志记录策略，流式数据传输时，可设置 `steps`，非流式传输为 "epoch"。
24. `--logging_steps 100`：如果训练日志记录的策略为 `steps`，这里指定多少步记录一次。
25. `--log_level "info"`：指定日志级别（未知作用）。
26. `--save_strategy "steps"`：checkpoint 保存策略，流式传输下设置为 "steps"。
27. `--save_steps x`：如果 checkpoint 保存策略为 steps，则每隔 x 步保存一次。
28. `--save_total_limit 10`：checkpoint 保存的最大数量
29. `--eval_strategy "steps"`：评估策略，流式传输下设置为 "steps"。
30. `--eval_steps x`：如果评估策略为 "steps"，则每隔 x 步评估一次。
31. `--seed 42`：随机数种子，如果两次运行种子相同，则应该能够复现。
32. `--disable_tqdm false`：是否禁用 tqdm，设置为 false 为启用
33. `--ddp_find_unused_parameters false`：DDP 找到未使用的参数（未知作用）。
34. `--block_size 512`：block 大小，类似于语言模型的上下文大小。
35. `--patch_size 32`：每个 block 内的字节数量。
36. `--step_size x`：步长，这个步长指的是，如果一个样本的字节数量大于 `block_size * patch_size`，则会在该样本上从头开始按此步长向后移动 x 个字节，产生一个新的样本，直至到尾部。
37. `--overwrite_output_dir true`：是否覆盖输出目录（不会覆盖目录内的 hugging face 日志）。
38. `--run_name "${output_model}"`：运行名称（未知作用）。
39. `--bf16`：使用 BF16 浮点数格式（未知作用）。
40. `--bf16_full_eval`：完全评估时，使用 BF16 浮点数格式（未知作用）。
41. `--gradient_checkpointing`：使用梯度检查点（未知作用）。
42. `--deepspeed ./ds_config_zero1.json`：指定 DeepSpeed 配置文件。
43. `--ignore_data_skip`：忽略数据跳过（未知作用）。
44. `--ddp_timeout 18000000`：DDP 超时时间（未知作用）。
45. `--trust_remote_code`：信任远程代码（未知作用）。
46. `--resume_from_checkpoint ${output_model}/checkpoint-18000`：从指定 checkpoint 开启训练。
47. `| tee -a "${output_model}/train.log"`：将输出追加到日志文件（主要是 hugging face 日志）。
