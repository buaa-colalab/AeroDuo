<div align="center">
<h1>AeroDuo: Aerial Duo for UAV-based Vision and Language Navigation</h1>

<image src="assets/teaser.png" width="100%">
<a href="https://arxiv.org/abs/2508.15232"><img src='https://img.shields.io/badge/arXiv-AeroDuo-red' alt='Paper PDF'></a>
<a href='https://rey-nard.github.io/AeroDuo_project/'><img src='https://img.shields.io/badge/Project_Page-AeroDuo-green' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env'><img src='https://img.shields.io/badge/Env-TRAVEL-blue'></a>
<a href='https://modelscope.cn/datasets/Reynard/HaL-13k/files'><img src='https://img.shields.io/badge/Dataset-HaL13k-blue'></a>
<a href='https://huggingface.co/datasets/salome1023/HaL-13k_testset'><img src='https://img.shields.io/badge/Dataset-HaL13k_testset-blue'></a>
</div>

# Contents

- [Introduction](#introduction)
- [News](#news)
- [Requirements and Installation](#requirements)
  - [Python Environment](#python-environment)
  - [Data](#data)
  - [Model](#model)
  - [Simulator environments](#simulator-environments)
- [Train](#train)
- [Evaluation](#eval)


# Introduction
This work presents **_AeroDuo: Aerial Duo for UAV-based Vision and Language Navigation_**. We introduce a dual-altitude collaborative framework, a dual-altitude VLN dataset, and a multimodal system for autonomous UAV flight.

# 📰News <a id="news"></a>
**2026-02-06:**  The train scripts and the complete HaL-13k are released.

**2025-12-05:**  Paper, project page, code, testset data, envs and models are all released.


# 🛠️ Requirements and Installation<a id="requirements"></a>

## Python Environment

### Create `aeroduo` environment

```bash
conda create -n aeroduo python=3.9 -y
conda activate aeroduo
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchrl==0.4.0
```

### Install other dependencies listed in the requirements file

```bash
pip install -r requirement.txt
```

Additionally, to ensure compatibility with the AirSim Python API, apply the fix mentioned in the [AirSim issue](https://github.com/microsoft/AirSim/issues/3333#issuecomment-827894198)

## Data
Download the dataset from [here](https://modelscope.cn/datasets/Reynard/HaL-13k/files), and arrange the files into the following hierarchy:
```
├───data
|   ├── HaL-13k
|   │   ├── Carla_Town05
|   │   ├── ModularPark
|   │   ├── NewYorkCity
|   │   ├── ...
```

Currently, only the test set is provided, but we will release the complete dataset in the future.

## Model

### GroundingDINO

Download the GroundingDINO model from the link [groundingdino_swint_ogc.pth](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth), and place the file in the directory `utils/GroundingDINO/`.

### PilotLLM

First download the pretrained weights from [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main), and then download the task-specific LoRA weights [AeroDuo-PilotLLM](https://huggingface.co/salome1023/AeroDuo-PilotLLM) used in the paper.

To finetune the PilotLLM on HaL-13k，you also need to download the pretrained weights on auxiliary task [pretrained_Qwen2-VL-2B-Instruct](https://huggingface.co/Rey-nard/pretrained_Qwen2-VL-2B-Instruct/tree/main).

Please organize your directory in the following structure:
```
├───pilot_llm
|   ├── weights
|   │   ├── Qwen2-VL-2B-Instruct
|   │   ├── AeroDuo-PilotLLM
|   |   ├── pretrained_Qwen2-VL-2B-Instruct
```

## Simulator environments

We adopt the simulation environments from the [OpenUAV](https://github.com/prince687028/TravelUAV) platform. You can download the simulator environments for various maps from [here](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env), specifically utilizing `carla_town_envs` and `closeloop_envs`.

The file directory of environments is as follows:
```
├───envs
|   ├── carla_town_envs
|   │   ├── Town01
|   │   ├── Town02
|   │   ├── Town03
|   │   ├── ...
|   ├── closeloop_envs
|   │   ├── Engine
|   │   ├── ModularEuropean
|   │   ├── ModularEuropean.sh
|   │   ├── ModularPark
|   │   ├── ModularPark.sh
|   │   ├── ...
```
# 🚀 Train <a id="train"></a>
We provide a shell script to finetune the PilotLLM on the HaL-13k dataset.

## 1. Configuration
Before running the `train.sh` script, ensure that the distributed training settings in `pilot_llm/default_deepspeed.yaml` match your situation. You can adjust parameters such as the `num_processes` to decide the number of gpu to use (default is 4).

## 2. Start Training
Run the following command to start the training process:

```bash
bash shell_scripts/train.sh
```

# ✅ Evaluation <a id="eval"></a>

## 1. Setup simulator env server

Before running the simulations, ensure that the AirSim environment server is properly configured.

> Update the env executable paths`env_exec_path_dict` relative to `root_path` in `AirVLNSimulatorServerTool.py`.

```bash
python airsim_plugin/AirVLNSimulatorServerTool.py --port 50000
```

## 2. Run close-loop simulation

Once the simulator server is running, you can execute the dagger or evaluation script.

```bash
# Eval
bash shell_scripts/eval.sh
```

## 3. Calculate final metrics

After evaluation, parse the output directory to compute metrics (SR, OSR, CR, SPL, SST, APL, NE) across all subsets:

```bash
python result_parser.py --result_dir ./output --testset data/test_unseen_new.json
```

The script prints a table with results for **Full**, **UM** (Unseen Map), and **UO** (Unseen Object) subsets.


To evaluate your own finetune result, replace the `llm_checkpoint_path` in the `eval.sh` with your own checkpoint path.

💡 **Performance Tip**:
The user could consider setting --use_a_star in `eval.sh` to False (default is True). The current internal A* algorithm implementation is unoptimized and may significantly result in slow execution speeds.

# 📆 TODO <a name="todos"></a>
- [x] Release the code and the testset of HaL-13k.
- [x] Release the train scripts and the complete HaL-13k.
- [ ] Optimize the A* algorithm.

# Paper

If you find this project useful, please consider citing:
```bibtex
    @inproceedings{wu2025aeroduo,
      title     = {AeroDuo: Aerial Duo for UAV-based Vision and Language Navigation},
      author    = {Wu, Ruipu and Zhang, Yige and Chen, Jinyu and Huang, Linjiang and Zhang, Shifeng and Zhou, Xu and Wang, Liang and Liu, Si},
      booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
      pages     = {2576--2585},
      year      = {2025}
    }
```

# Acknowledgement

This repository is partly based on [TravelUAV](https://github.com/prince687028/TravelUAV) and [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) repositories.
