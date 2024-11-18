# BBT-Neutron

## 模型架构

<p align="center">
<div style="display: flex; justify-content: center;">
    <img src="./figures/BBT_model_arch.png?raw=true" style="height:300px; width:auto; margin-right:10px">
</div>
</p>

## 数据处理

分两种方案：
- 流式启动训练
- 直接非流式启动训练
- 预处理后，非流式启动训练

### 流式启动训练

如果没做数据预处理，则在训练代码中，使用下列代码，训练时预处理。分为训练时流式处理启动训练和训练时非流式处理启动训练。
```py
    # Load datasets
    logger.info("start load datasets")
    with training_args.main_process_first(desc="dataset map"):
        datasets = load_and_prepare_datasets(task_type=data_args.task_type,
                                             train_dir=data_args.train_dir,
                                             validation_dir=data_args.validation_dir,
                                             test_dir=None,
                                             block_size=data_args.block_size,
                                             patch_size=data_args.patch_size,
                                             step_size=data_args.step_size,
                                             pad_id=0,
                                             streaming=data_args.streaming,
                                             cache_dir=None,
                                             num_proc=data_args.preprocessing_num_workers)
        # datasets = get_arrow_dataset(data_args.data_dir)
    logger.info("end load datasets")
```

在训练主要加载数据代码中，`--streaming true`。指定数据目录`--train_dir`和`--validation_dir`。
该状态，代码会使用`data_utils.py`的`load_and_prepare_dataset`，实时使用数据处理的方式进行训练。
此时数据每次加载一条处理一条，在多机继续训练时`--ignore_data_skip true`，否则会导致需要大量时间处理已经训练过的数据

### 直接非流式启动训练

`--streaming false` dataset map函数将数据处理缓存到cache下。注意处理时间，缓存空间等问题

### 预处理后，非流式启动训练

比较推荐先做数据预处理，后训练代码中使用下列代码，加载预处理的数据进行使用。
```py
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
```
此时需要指定`--streaming true`，指定数据目录`--data_dir`

### 数据预处理

参考见`data_utils.py`的```if __name__=="__main__"```:方法，预先处理数据，使用`preprocess_dataset`。

该部分是字节型预处理，使用方法详见`data_utils.py`代码以及`BBT-Neutron-SimpleGuide.md`相关使用说明。

其中数据预处理需要注意，是使用pad模式还是拼接模式。即需要将所有数据拼接成一个长字符串，然后再按照模型长度做chunk，再舍弃最后那个不足模型长度的部分。

或者全部直接做chunk，不足的pad到模型长度。一般本代码预训练中用前者，舍弃pad加快速度，微调时再用单个doc做pad免除多个doc之间干扰。

前者的情况建议在语料`bos`，`eos`添加特殊token。

## 模型

包含v1和v2版本需要根据需求切换，v1版本带patch总体计算速度更快，v2是纯字节模型。由于patch存在，v2模型兼容hf模型功能。

使用时修改```model/__init__.py```，指定对对应的模型代码加载。

配套的`pretrain.py`也需要将对应`BinBBTConfig`的`config.auto_map`填写这正确（用于保存的权重顺利加载）。

字节模型的tokenizer完全不同，本项目开发了字节tokenizer，`tokenization_binbbt.py`有待更完善的功能测试。

## 环境

在docker文件夹下有几乎所有的依赖要求，请务必对应所有的依赖库，今后做好环境管理和环境升级。本项目主要兼容hf模型和相关接口，所以大部分为hf模型需要的以来组件。

config文件夹下有训练需要的环境参数，对应dockerfile。

## 训练，推理，测评

使用`pretrain.sh`训练，`generate.sh`对话，`inference.sh`推理，`eval_clm.sh`测评。

训练，对话，推理代码有很大相似度，如果修改请全局考虑。都使用了hf model，hf trainer等通用方案。

推理可以不使用deepspeed，即删除`--deepspeed`。如果使用deepspeed inference推理则必须要用zero3做模型并行。该功能现在版本已经正常。

## 其他参数设置方法

见使用者贾兴隆制作的 `BBT-Neutron-SimpleGuide.md`

peft功能和v1模型的generate功能，尚未完善。

### 启动命令

启动训练。参数详见`BBT-Neutron-SimpleGuide.md`和[hf trainer官方参考](https://huggingface.co/docs/transformers/main_classes/trainer)。修改`model/__init__.py`选择正确模型，设置好`configuration_binbbt`的模型参数。
```bash
cd train/pretrain
bash pretrain.sh
```

启动测评。修改`--config_file`配置参数，`--model_args`模型参数，`--tasks`测评任务，`--output_path`输出路径，`--num_fewshot`。
```bash
cd evaluate
bash eval_clm.sh
```

启动推理。
```bash
cd inference
bash inference.sh
```

启动生成。需要修改`generate.py`文件，确定生成数据长度和模型加载方式，需要预处理生成文本。
```bash
cd inference
bash generate.sh
```

启动数据预处理。由于需要用到tokenizer，所以需要`export PYTHONPATH="/path/to/bbt-neutron:${PYTHONPATH}"`声明项目路径
```bash
cd data
export PYTHONPATH="/path/to/bbt-neutron:${PYTHONPATH}"
python3 data_utils.py
```
