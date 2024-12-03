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
  <a href="BBT-Neutron/LICENSE at main · ssymmetry/bbt-neutron" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="BBT-Neutron/LICENSE-MODEL at main · ssymmetry/bbt-neutron" style="margin: 2px;">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="#2-model-architecture">Model Architecture</a> |
  <a href="#3-data-processing">Data Processing</a> |
  <a href="#5-environment">Environment</a> |
  <a href="#6-training-inference-and-evaluation">Training, Inference and Evaluation</a> |
  <a href="#7-quick-start">Quick Start</a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/pdf/2412.00129"><b>Paper Link</b>👁️</a>
</div>



# BBT-Neutron: A Task-agnostic Large Language Model Architecture Specifically Designed for Scientific Tasks

## 1. Introduction

For decades, researchers have developed task-specific models to address scientific challenges across diverse disciplines. Recently, large language models (LLMs) have shown enormous capabilities in handling general tasks; however, these models encounter difficulties in addressing real-world scientific problems, particularly in domains involving large-scale numerical data analysis, such as experimental high- energy physics. This limitation is primarily due to BPE tokenization’s inefficacy with numerical data. In this paper, we propose a task-agnostic architecture, BBT- Neutron, which employs a binary tokenization method to facilitate pretraining on a mixture of textual and large-scale numerical experimental data. We demonstrate the application of BBT-Neutron to Jet Origin Identification (JoI), a critical categorization challenge in high-energy physics that distinguishes jets from various quarks or gluons. Our results indicate that BBT-Neutron achieves comparable performance to state-of-the-art task-specific JoI models. Furthermore, we exam- ine the scaling behavior of BBT-Neutron’s performance with increasing data size, suggesting the potential for BBT-Neutron to serve as a foundational model for particle physics data analysis, with possible extensions to a broad spectrum of scientific computing applications for Big Science experiments, industrial manufacturing and spacial computing.

## 2. Model Architecture

<div align="center" style="display: flex; justify-content: center;">
    <img src="./figures/BBT_model_arch.png?raw=true" style="height:600px; width:auto; margin-right:10px">
</div>

## 3. Data Processing

Available options：
- streaming training mode
- direct non-streaming training mode
- non-streaming training mode after preprocessing

### Streaming Training Mode

If data preprocessing has not been done, use the following code in your training script to preprocess the data during training. This can be divided into streaming processing and non-streaming processing during training.

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

Please set `--streaming true` in the datasets loading code, and specify the data directories via `--train_dir` and `--validation_dir`.

In this state, the code will use function `load_and_prepare_dataset()` in `data_utils.py` to do real-time data pre-processing during the training period. Each sample will be loaded and processed one at a time. When continuing training on multiple machines, use `--ignore_data_skip true` to avoid spending a lot of time processing already trained data.

### Direct Non-Streaming Training Mode

Please set `--streaming false`. Under this mode, datasets will be cached and stored in the datasets map function. Be careful of the time cost and memory space!

### Non-Streaming Training Mode After Preprocessing

It's strongly recommended that you preprocess the data first, and then load preprocessed data using the following code in your training script.

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

You need to set `--streaming true`，and `--data_dir` to your specific input directory.

### Data Pre-processing

For reference, see the `main()` function in `data_utils.py` for more detail. The code will use `preprocess_dataset` function for data preprocessing.

This part involves byte-level preprocessing. Details can be found in `data_utils.py` and related usage instructions in `BBT-Neutron-SimpleGuide.md`.

There are two options when you do data preprocessing, namely, the padding mode or the concatenation mode. 

`The concatenation mode:` All the datasets are concatenated into a long string, and then the string is divided into chunks according to the model's length. The last chunk is discarded unless its length is equal to the model's length.

`The padding mode:` Each dataset is directly divided into chunks without concatenation. We use padding technique to process the last chunk of each dataset, instead of discarding it, by adding special tokens to the end of the chunk.

Generally, in the pre-training stage, we use concatenation mode to discard padding and speed up the process. While in the fine-tuning stage, we prefer to use padding mode in order to avoid the interference between multiple documents.

In the concatenation mode, we recommend that you add special tokens such as `<bos>` and `<eos>` in your training corpus. 

## 4. Model

Two model versions, namely v1 and v2, are available. V1 model is a patch model, and the overall time cost for training is lower. V2 model is a byte model which is compatible with the HuggingFace model interface, while v1 model is not.

Modify ```model/__init__.py``` to load the model version you need.

In addition, you need to correctly set `config.auto_map` corresponding to `BinBBTConfig` in `pretrain.py`, to ensure your pre-trained model weights can be loaded successfully.

In this project, tokenizer for the byte model is newly developed, which is compeletely different from the traditional tokenizer trained with BPE method. 

A more thorough functional test for `tokenization_binbbt.py` is needed.

## 5. Environment

Almost all dependency requirements are located in the `docker/` folder. Please ensure that all dependent libraries are aligned accordingly, and make efforts in managing and upgrading the environment in the future. This project is compatible with the HuggingFace models and related interfaces, so most dependencies are reuqired by the HuggingFace model.

The `config/` floder contains environmental parameters needed for training, which correspond to the dockerfile.

## 6. Training, Inference and Evaluation

Please use `pretrain.sh` for training, `generate.sh` for dialogue generation, `inference.sh` for inference and `eval_clm.sh` for evaluation。

There is a high degree of similarity between the training, conversation and inference code, please take a global consideration when you want to modify the code. Common solutions such as HuggingFace model and HuggingFace Trainer are used in all of the cases.

Deepspeed is not mandatory in inference, which means you can remove `--deepspeed` in the startup script. If you want to use deepspeed for inference, make sure you use `zero3` configuration setting to include model parallelism. This feature is already available in the current version.

## 7. Quick Start

If you want to use our project for a jet tagging task, please follow the steps below for a quick start.

1. Datasets 

    An example of the datasets used in the jet tagging task is shown below:
```jsonl
{"data": "0: -1, 7.982902, (-7.899555, 0.408736, -1.066399), 2.077302, 2.068143, -0.098329, -0.213443, -2.259137, -2.381407, 0.235003, 0.024568, 0.048371, -0.042774, 0.049214, charged pion. 1: -1, 6.685446, (-6.462185, -0.339102, -1.605216), 1.899933, 1.867342, 0.012845, -0.109320, -2.459938, -2.558776, 0.110073, -0.075995, 0.049435, 0.110177, 0.050115, charged kaon. 2: 1, 4.962568, (-4.854748, 0.267230, -0.983670), 1.601923, 1.581470, -0.031780, -0.216737, -2.745810, -2.856786, 0.219054, 0.057629, 0.051062, 0.051798, 0.051452, charged pion. 3: 0, 4.401175, (-4.365309, -0.099968, -0.551752), 1.481872, 1.473951, -0.106711, -0.138851, -2.853329, -2.976837, 0.175119, 0.000000, 0.000000, 0.000000, 0.000000, photon. 4: 0, 4.395812, (-4.382381, 0.044969, -0.340397), 1.480652, 1.477645, -0.155146, -0.172008, -2.849635, -2.978057, 0.231640, 0.000000, 0.000000, 0.000000, 0.000000, photon. 5: -1, 4.005426, (-3.877740, -0.056291, -0.991927), 1.387650, 1.355358, 0.020325, -0.147232, -2.971922, -3.071059, 0.148628, -0.048030, 0.052246, 0.063851, 0.052600, charged pion. 6: 1, 3.805775, (-3.662414, 0.301091, -0.980059), 1.336520, 1.301490, 0.030896, -0.243774, -3.025790, -3.122189, 0.245724, 0.053499, 0.052695, 0.053155, 0.053051, charged pion. 7: 0, 3.426863, (-3.365381, 0.404382, -0.504060), 1.231645, 1.220709, -0.084573, -0.281333, -3.106572, -3.227064, 0.293770, 0.000000, 0.000000, 0.000000, 0.000000, photon. 8: 0, 3.352921, (-3.278398, 0.056086, -0.700743), 1.209832, 1.187501, -0.020619, -0.178853, -3.139779, -3.248877, 0.180038, 0.000000, 0.000000, 0.000000, 0.000000, photon. 9: 1, 2.926103, (-2.616569, 0.069915, -0.911266), 1.073671, 0.962221, 0.108730, -0.188461, -3.365059, -3.385038, 0.217577, -0.270038, 0.055539, 0.192414, 0.056045, proton. 10: 0, 2.664285, (-1.337302, -2.149693, -0.829975), 0.979936, 0.928895, 0.089488, 0.852545, -3.398385, -3.478773, 0.857229, 0.000000, 0.000000, 0.000000, 0.000000, photon. 11: 0, 2.310138, (-2.300164, -0.031352, -0.212132), 0.837307, 0.833073, -0.140652, -0.148118, -3.494207, -3.621402, 0.204259, 0.000000, 0.000000, 0.000000, 0.000000, photon. 12: 1, 2.307637, (-0.129767, -0.752603, -2.173122), 0.836224, -0.269569, 1.535680, 1.238304, -4.596849, -3.622485, 1.972742, -0.174068, 0.070675, 0.164366, 0.073021, charged pion. 13: 0, 2.294021, (-2.285264, 0.029542, -0.198057), 0.830306, 0.826565, -0.146187, -0.174674, -3.500715, -3.628403, 0.227775, 0.000000, 0.000000, 0.000000, 0.000000, photon. 14: 0, 2.176314, (-2.152707, 0.127555, -0.293127), 0.777632, 0.768479, -0.097225, -0.220931, -3.558802, -3.681077, 0.241378, 0.000000, 0.000000, 0.000000, 0.000000, photon. 15: -1, 2.159010, (-0.841519, 0.001196, -1.752945), 0.769650, -0.172546, 1.247442, -0.163168, -4.499826, -3.689059, 1.258068, 0.318025, 0.069354, -0.397044, 0.071495, proton. 16: 0, 1.953062, (-1.899484, -0.078596, -0.447475), 0.669398, 0.642438, 0.000516, -0.120393, -3.684843, -3.789311, 0.120394, 0.000000, 0.000000, 0.000000, 0.000000, photon. 17: 0, 1.936481, (-1.843615, -0.351870, -0.476689), 0.660873, 0.629618, 0.018585, 0.026844, -3.697662, -3.797836, 0.032649, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 18: 1, 1.909002, (-0.793892, -1.595945, -0.668957), 0.646581, 0.578017, 0.134259, 0.947449, -3.749264, -3.812129, 0.956914, -0.108346, 0.060004, 0.114991, 0.060692, charged pion. 19: 0, 1.802877, (-0.916724, -1.530938, -0.257320), 0.589384, 0.579093, -0.089030, 0.869513, -3.748187, -3.869325, 0.874059, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 20: 1, 1.799525, (-0.458576, -0.713431, -1.580992), 0.587523, -0.164756, 1.148441, 0.837769, -4.492036, -3.871186, 1.421539, 0.245924, 0.069312, 0.223021, 0.071569, charged pion. 21: 1, 1.784490, (-1.749027, -0.175007, -0.274235), 0.579133, 0.564040, -0.077351, -0.062020, -3.763240, -3.879576, 0.099144, -0.060723, 0.059966, 0.058150, 0.060117, charged pion. 22: 1, 1.715898, (-1.641326, -0.320969, -0.357565), 0.539937, 0.514269, -0.020533, 0.031370, -3.813011, -3.918772, 0.037493, -0.133285, 0.060781, 0.095457, 0.061044, charged pion. 23: 0, 1.570416, (-1.480101, 0.401845, -0.337678), 0.451340, 0.427671, -0.014305, -0.426855, -3.899609, -4.007369, 0.427095, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 24: -1, 1.486897, (-0.457783, -0.698566, -1.222220), 0.396691, -0.180084, 0.941541, 0.828944, -4.507364, -4.062018, 1.254451, 0.128159, 0.070086, 0.310163, 0.073498, charged pion. 25: -1, 1.414431, (-0.309921, -0.555473, 1.255601), 0.346727, -0.452426, -1.664667, 0.900130, -4.779706, -4.111982, 1.892445, 0.029776, 0.072301, -0.188226, 0.072500, charged pion. 26: -1, 1.371773, (-0.769531, 0.073952, -1.124560), 0.316104, -0.257378, 0.936606, -0.257552, -4.584658, -4.142605, 0.971373, 0.053427, 0.069742, 0.056873, 0.070574, charged pion. 27: 0, 1.360614, (-1.321092, 0.075450, -0.316692), 0.307936, 0.280087, 0.004363, -0.218797, -4.047194, -4.150773, 0.218841, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron. 28: -1, 1.226438, (-0.412902, -0.973076, -0.378344), 0.204114, 0.055487, 0.117952, 1.007749, -4.271793, -4.254595, 1.014628, -0.129970, 0.065141, -0.091094, 0.065708, charged kaon. 29: -1, 1.190959, (-0.008252, -0.377224, -1.120954), 0.174759, -0.974676, 1.576457, 1.387178, -5.301956, -4.283950, 2.099876, -0.301847, 0.085622, -0.273366, 0.086198, charged pion. 30: 1, 1.094247, (-0.386346, -0.986615, -0.234999), 0.090066, 0.057856, -0.012729, 1.035816, -4.269424, -4.368642, 1.035894, -0.138490, 0.064938, -0.138975, 0.065153, charged pion. 31: 0, 1.054287, (-1.052853, 0.016745, -0.052341), 0.052864, 0.051630, -0.183052, -0.177651, -4.275650, -4.405845, 0.255084, 0.000000, 0.000000, 0.000000, 0.000000, photon. 32: 0, 0.992163, (-0.458974, -0.850251, -0.225393), -0.007868, -0.034362, -0.001531, 0.914063, -4.361642, -4.466577, 0.914065, 0.000000, 0.000000, 0.000000, 0.000000, photon. 33: 0, 0.913373, (-0.502837, -0.727852, -0.227237), -0.090611, -0.122558, 0.021383, 0.804490, -4.449838, -4.549320, 0.804774, 0.000000, 0.000000, 0.000000, 0.000000, photon. 34: 0, 0.814941, (-0.802700, 0.030826, -0.137299), -0.204640, -0.219038, -0.062639, -0.200132, -4.546318, -4.663349, 0.209705, 0.000000, 0.000000, 0.000000, 0.000000, photon. 35: 0, 0.786306, (-0.767984, -0.024626, -0.166947), -0.240409, -0.263472, -0.017141, -0.129692, -4.590753, -4.699118, 0.130820, 0.000000, 0.000000, 0.000000, 0.000000, photon. 36: 0, 0.767866, (-0.419870, -0.599481, -0.232270), -0.264140, -0.312120, 0.079518, 0.798062, -4.639400, -4.722849, 0.802013, 0.000000, 0.000000, 0.000000, 0.000000, photon. 37: 0, 0.646095, (-0.635767, 0.062700, -0.096475), -0.436809, -0.448084, -0.082293, -0.260050, -4.775364, -4.895518, 0.272760, 0.000000, 0.000000, 0.000000, 0.000000, photon. 38: 0, 0.500555, (-0.025322, -0.282856, -0.412197), -0.692038, -1.258826, 0.934797, 1.319764, -5.586107, -5.150747, 1.617289, 0.000000, 0.000000, 0.000000, 0.000000, neutral hadron.", "label": 0}
```

The sample is organized in a json format, with the key 'data' containing all the information of the jet and the key 'label' representing the category of the jet. The numbers 0,1,2,3 ... followed by a colon, represent the indices of the final state particles inside the jet. Information after the colon represents different attributes of the final state particle, including the particle's momentum, energy, etc. The list of variables used in this task and their corresponding definitions are shown in the figure below:

<div align="center">
  <img src="./figures/input_var_list.png?raw=true" width="60%" alt="List of Variables" />
</div>

2. Data pre-processing

```bash
cd example
python preprocess_data.py
```

parameter configuration

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

3. Training

```bash
cd example
bash run_jet_tagging_train.sh
```

key parameter configuration

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

## 8. Other Parameter Settings

Please refer to `BBT-Neutron-SimpleGuide.md` written by Xinglong Jia for more details.

The peft function and generate function of the v1 model are still in development.

### Startup Commands

Start training with the following commands. Please refer to `BBT-Neutron-SimpleGuide.md` and [HuggingFace Official Documents](https://huggingface.co/docs/transformers/main_classes/trainer) for the detailed parameter settings. Modify `model/__init__.py` to select the correct model you need, and also don't forget to properly set the model parameters in `configuration_binbbt.py`. 

```bash
cd train/pretrain
bash pretrain.sh
```

Start evaluation with the following commands. You are free to modify `--config_file`, `--model_args`, `--tasks`, `--output_path` and `num_fewshot` according to your preference. 

```bash
cd evaluate
bash eval_clm.sh
```

Start inference with the following commands.

```bash
cd inference
bash inference.sh
```

Start conversation generation with the following commands. 

```bash
cd inference
bash generate.sh
```

Start data pre-processing with the following commands. Please declare the project path with: `export PYTHONPATH="/path/to/bbt-neutron:${PYTHONPATH}"`, since data pre-processing will use tokenizer.

```bash
cd data
export PYTHONPATH="/path/to/bbt-neutron:${PYTHONPATH}"
python3 data_utils.py
```
