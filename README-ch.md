<div align="center">
  <img src="./figures/ssymmetry-logo.png?raw=true" width="60%" alt="Ssymmetry" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.ssymmetry.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="./figures/super_symmetry-group-blue.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://www.ssymmetry.com" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="./figures/big_bang_transformer-neutron-blue.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


<div align="center" style="line-height: 1;">
  <a href="https://github.com/supersymmetry-technologies/bbt-neutron/blob/main/LICENSE" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/supersymmetry-technologies/bbt-neutron/blob/main/LICENSE-MODEL" style="margin: 2px;">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="#2-model-architecture">模型架构</a> |
  <a href="#3-data-processing">数据处理</a> |
  <a href="#5-environment">系统环境</a> |
  <a href="#6-training-inference-and-evaluation">训练,推理和评测</a> |
  <a href="#7-quick-start">快速开始</a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/pdf/2412.00129"><b>论文链接</b>👁️</a>
</div>

# BBT-Neutron: 一个专为科学领域设计的多任务大语言模型

## 1. 简介

几十年来，研究人员开发了特定任务的模型，以解决跨不同学科的科学挑战。最近，大型语言模型（LLM）在处理通用任务方面表现出了巨大的能力；然而，这些模型在解决现实世界中的科学问题，特别是在涉及大规模数值数据分析的领域（如实验高能物理）时遇到了困难。这一局限性主要是由于BPE标记化方法在处理数值数据时效果不佳。在本文中，我们提出了一种与任务无关的架构BBT-Neutron，该架构采用二进制标记化方法来促进对文本和大规模数值实验数据的混合进行预训练。我们展示了BBT-Neutron在喷注起源识别（Jet Origin Identification，JoI）中的应用，这是高能物理中的一个关键分类挑战，用于区分来自不同夸克或胶子的喷注。我们的结果表明，BBT-Neutron的性能与最先进的特定任务JoI模型相当。此外，我们还研究了BBT-Neutron性能随数据量增加的扩展行为，这表明BBT-Neutron有潜力成为粒子物理数据分析的基础模型，并可能扩展到大型科学实验、工业制造和空间计算等广泛领域的科学计算应用。

## 2. 模型架构

<div align="center" style="display: flex; justify-content: center;">
    <img src="./figures/BBT_model_arch.png?raw=true" style="height:600px; width:auto; margin-right:10px">
</div>

## 3. 数据处理

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

## 4. 模型

包含v1和v2版本需要根据需求切换，v1版本带patch总体计算速度更快，v2是纯字节模型。由于patch存在，v2模型兼容hf模型功能。

使用时修改```model/__init__.py```，指定对对应的模型代码加载。

配套的`pretrain.py`也需要将对应`BinBBTConfig`的`config.auto_map`填写这正确（用于保存的权重顺利加载）。

字节模型的tokenizer完全不同，本项目开发了字节tokenizer，`tokenization_binbbt.py`有待更完善的功能测试。

## 5. 环境

在docker文件夹下有几乎所有的依赖要求，请务必对应所有的依赖库，今后做好环境管理和环境升级。本项目主要兼容hf模型和相关接口，所以大部分为hf模型需要的依赖组件。

config文件夹下有训练需要的环境参数，对应dockerfile。

## 6. 训练，推理，测评

使用`pretrain.sh`训练，`generate.sh`对话，`inference.sh`推理，`eval_clm.sh`测评。

训练，对话，推理代码有很大相似度，如果修改请全局考虑。都使用了hf model，hf trainer等通用方案。

推理可以不使用deepspeed，即删除`--deepspeed`。如果使用deepspeed inference推理则必须要用zero3做模型并行。该功能现在版本已经正常。

## 7. 快速开始

如果您希望使用我们的项目来执行一个粒子物理jet标注任务，请参照下面的步骤：

1. 数据集

    在粒子物理jet标注任务中，一个典型的数据样本的示例如下：
```jsonl
{"data": "0: -1, 7.982902, (-7.899555, 0.408736, -1.066399), 2.077302, 2.068143, -0.098329, -0.213443, -2.259137, -2.381407, 0.235003, 0.024568, 0.048371, -0.042774, 0.049214, charged pion. 1: -1, 6.685446, (-6.462185, -0.339102, -1.605216), 1.899933, 1.867342, 0.012845, -0.109320, -2.459938, -2.558776, 0.110073, -0.075995, 0.049435, 0.110177, 0.050115, charged kaon. 2: 1, 4.962568, (-4.854748, 0.267230, -0.983670), 1.601923, 1.581470, -0.031780, -0.216737, -2.745810, -2.856786, 0.219054, 0.057629, 0.051062, 0.051798, 0.051452, charged pion. 3: 0, 4.401175, (-4.365309, -0.099968, -0.551752), 1.481872, 1.473951, -0.106711, -0.138851, -2.853329, -2.976837, 0.175119, 0.000000, 0.000000, 0.000000, 0.000000, photon. 4: 0, 4.395812, (-4.382381, 0.044969, -0.340397), 1.480652, 1.477645, -0.155146, -0.172008, -2.849635, -2.978057, 0.231640, 0.000000, 0.000000, 0.000000, 0.000000, photon. 5: -1, 4.005426, (-3.877740, -0.056291, -0.991927), 1.387650, 1.355358, 0.020325, -0.147232, -2.971922, -3.071059, 0.148628, -0.048030, 0.052246, 0.063851, 0.052600, charged pion. 6: 1, 3.805775, (-3.662414, 0.301091, -0.980059), 1.336520, 1.301490, 0.030896, -0.243774, -3.025790, -3.122189, 0.245724, 0.053499, 0.052695, 0.053155, 0.053051, charged pion. 7: 0, 3.426863, (-3.365381, 0.404382, -0.504060), 1.231645, 1.220709, -0.084573, -0.281333, -3.106572, -3.227064, 0.293770, 0.000000, 0.000000, 0.000000, 0.000000, photon. 8: 0, 3.352921, (-3.278398, 0.056086, -0.700743), 1.209832, 1.187501, -0.020619, -0.178853, -3.139779, -3.248877, 0.180038, 0.000000, 0.000000, 0.000000, 0.000000, photon. 9: 1, 2.926103, (-2.616569, 0.069915, -0.911266), 1.073671, 0.962221, 0.108730, -0.188461, -3.365059, -3.385038, 0.217577, -0.270038, 0.055539, 0.192414, 0.056045, proton. 10: 0, 2.664285, (-1.337302, -2.149693, -0.829975), 0.979936, 0.928895, 0.089488, 0.852545, -3.398385, -3.478773, 0.857229, 0.000000, 0.000000, 0.000000, 0.000000, photon. 11: 0, 2.310138, (-2.300164, -0.031352, -0.212132), 0.837307, 0.833073, -0.140652, -0.148118, -3.494207, -3.621402, 0.204259, 0.000000, 0.000000, 0.000000, 0.000000, photon. 12: 1, 2.307637, (-0.129767, -0.752603, -2.173122), 0.836224, -0.269569, 1.535680, 1.238304, -4.596849, -3.622485, 1.972742, -0.174068, 0.070675, 0.164366, 0.073021, charged pion. 13: 0, 2.294021, (-2.285264, 0.029542, -0.198057), 0.830306, 0.826565, -0.146187, -0.174674, -3.500715, -3.628403, 0.227775, 0.000000, 0.000000, 0.000000, 0.000000, photon. 14: 0, 2.176314, (-2.152707, 0.127555, -0.293127), 0.777632, 0.768479, -0.097225, -0.220931, -3.558802, -3.681077, 0.241378, 0.000000, 0.000000, 0.000000, 0.000000, photon. 15: -1, 2.159010, (-0.841519, 0.001196, -1.752945), 0.769650, -0.172546, 1.247442, -0.163168, -4.499826, -3.689059, 1.258068, 0.318025, 0.069354, -0.397044, 0.071495, proton. 16: 0, 1.953062, (-1.899484, -0.078596, -0.447475), 0.669398, 0.642438, 0.000516, -0.120393, -3.684843, -3.789311, 0.120394, 0.000000, 0.000000, 0.000000, 0.000000, photon. 17: 0, 1.936481, (-1.843615, -0.351870, -0.476689), 0.660873, 0.629618, 0.018585, 0.026844, -3.697662, -3.797836, 0.032649, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 18: 1, 1.909002, (-0.793892, -1.595945, -0.668957), 0.646581, 0.578017, 0.134259, 0.947449, -3.749264, -3.812129, 0.956914, -0.108346, 0.060004, 0.114991, 0.060692, charged pion. 19: 0, 1.802877, (-0.916724, -1.530938, -0.257320), 0.589384, 0.579093, -0.089030, 0.869513, -3.748187, -3.869325, 0.874059, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 20: 1, 1.799525, (-0.458576, -0.713431, -1.580992), 0.587523, -0.164756, 1.148441, 0.837769, -4.492036, -3.871186, 1.421539, 0.245924, 0.069312, 0.223021, 0.071569, charged pion. 21: 1, 1.784490, (-1.749027, -0.175007, -0.274235), 0.579133, 0.564040, -0.077351, -0.062020, -3.763240, -3.879576, 0.099144, -0.060723, 0.059966, 0.058150, 0.060117, charged pion. 22: 1, 1.715898, (-1.641326, -0.320969, -0.357565), 0.539937, 0.514269, -0.020533, 0.031370, -3.813011, -3.918772, 0.037493, -0.133285, 0.060781, 0.095457, 0.061044, charged pion. 23: 0, 1.570416, (-1.480101, 0.401845, -0.337678), 0.451340, 0.427671, -0.014305, -0.426855, -3.899609, -4.007369, 0.427095, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 24: -1, 1.486897, (-0.457783, -0.698566, -1.222220), 0.396691, -0.180084, 0.941541, 0.828944, -4.507364, -4.062018, 1.254451, 0.128159, 0.070086, 0.310163, 0.073498, charged pion. 25: -1, 1.414431, (-0.309921, -0.555473, 1.255601), 0.346727, -0.452426, -1.664667, 0.900130, -4.779706, -4.111982, 1.892445, 0.029776, 0.072301, -0.188226, 0.072500, charged pion. 26: -1, 1.371773, (-0.769531, 0.073952, -1.124560), 0.316104, -0.257378, 0.936606, -0.257552, -4.584658, -4.142605, 0.971373, 0.053427, 0.069742, 0.056873, 0.070574, charged pion. 27: 0, 1.360614, (-1.321092, 0.075450, -0.316692), 0.307936, 0.280087, 0.004363, -0.218797, -4.047194, -4.150773, 0.218841, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 28: -1, 1.226438, (-0.412902, -0.973076, -0.378344), 0.204114, 0.055487, 0.117952, 1.007749, -4.271793, -4.254595, 1.014628, -0.129970, 0.065141, -0.091094, 0.065708, charged kaon. 29: -1, 1.190959, (-0.008252, -0.377224, -1.120954), 0.174759, -0.974676, 1.576457, 1.387178, -5.301956, -4.283950, 2.099876, -0.301847, 0.085622, -0.273366, 0.086198, charged pion. 30: 1, 1.094247, (-0.386346, -0.986615, -0.234999), 0.090066, 0.057856, -0.012729, 1.035816, -4.269424, -4.368642, 1.035894, -0.138490, 0.064938, -0.138975, 0.065153, charged pion. 31: 0, 1.054287, (-1.052853, 0.016745, -0.052341), 0.052864, 0.051630, -0.183052, -0.177651, -4.275650, -4.405845, 0.255084, 0.000000, 0.000000, 0.000000, 0.000000, photon. 32: 0, 0.992163, (-0.458974, -0.850251, -0.225393), -0.007868, -0.034362, -0.001531, 0.914063, -4.361642, -4.466577, 0.914065, 0.000000, 0.000000, 0.000000, 0.000000, photon. 33: 0, 0.913373, (-0.502837, -0.727852, -0.227237), -0.090611, -0.122558, 0.021383, 0.804490, -4.449838, -4.549320, 0.804774, 0.000000, 0.000000, 0.000000, 0.000000, photon. 34: 0, 0.814941, (-0.802700, 0.030826, -0.137299), -0.204640, -0.219038, -0.062639, -0.200132, -4.546318, -4.663349, 0.209705, 0.000000, 0.000000, 0.000000, 0.000000, photon. 35: 0, 0.786306, (-0.767984, -0.024626, -0.166947), -0.240409, -0.263472, -0.017141, -0.129692, -4.590753, -4.699118, 0.130820, 0.000000, 0.000000, 0.000000, 0.000000, photon. 36: 0, 0.767866, (-0.419870, -0.599481, -0.232270), -0.264140, -0.312120, 0.079518, 0.798062, -4.639400, -4.722849, 0.802013, 0.000000, 0.000000, 0.000000, 0.000000, photon. 37: 0, 0.646095, (-0.635767, 0.062700, -0.096475), -0.436809, -0.448084, -0.082293, -0.260050, -4.775364, -4.895518, 0.272760, 0.000000, 0.000000, 0.000000, 0.000000, photon. 38: 0, 0.500555, (-0.025322, -0.282856, -0.412197), -0.692038, -1.258826, 0.934797, 1.319764, -5.586107, -5.150747, 1.617289, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron.", "label": 0}
```

样本是json格式保存的，其中关键字'data'中包含了关于当前jet的所有信息，关键字'label'表示了这个jet的种类。数据0,1,2,3加上冒号，表示当前jet衰变的末态粒子的编号。冒号之后的信息包含了末态粒子的各种属性，例如粒子动量，能量等。在该任务中使用的各个物理学变量以及相关的定义列在下表中：

<div align="center">
  <img src="./figures/input_var_list.png?raw=true" width="60%" alt="List of Variables" />
</div>

2. 数据预处理

```bash
cd example
python preprocess_data.py
```

参数配置

```python
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
```

3. 训练

```bash
cd example
bash run_jet_tagging_train.sh
```

关键参数配置

```bash
    --task_type classifacation \
    --train_dir ./data/processed/train \
    --validation_dir ./data/processed/validation \
    --cache_dir ./data/cache \
    --streaming false \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir "${output_model}" \
```

## 8. 其他参数设置方法

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
