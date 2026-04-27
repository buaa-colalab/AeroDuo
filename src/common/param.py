import argparse
import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import transformers

from utils.CN import CN


@dataclass
class CommonArguments:
    project_prefix: str = field(
        default_factory=lambda: str(Path(str(os.getcwd())).parent.resolve()),
        metadata={"help": "project path"})
    run_type: str = field(
        default="train",
        metadata={"help": "run_type in [collect, train, eval]"})
    policy_type: str = field(
        default="cma", metadata={"help": "policy_type in [seq2seq, cma]"})
    collect_type: str = field(default="TF",
                              metadata={"help": "seq2seq in [TF, dagger, SF]"})
    name: str = field(default='default', metadata={"help": 'experiment name'})

    maxInput: int = field(default=500,
                          metadata={"help": "max input instruction"})
    maxWaypoints: int = field(default=50,
                              metadata={"help": 'max action sequence'})

    dagger_it: int = field(default=1)
    epochs: int = field(default=10)
    lr: float = field(default=0.00025, metadata={"help": "learning rate"})
    batchSize: int = field(default=8)
    trainer_gpu_device: int = field(default=0, metadata={"help": 'GPU'})

    Image_Height_RGB: int = field(default=224)
    Image_Width_RGB: int = field(default=224)
    Image_Height_DEPTH: int = field(default=256)
    Image_Width_DEPTH: int = field(default=256)

    inflection_weight_coef: float = field(default=1.9)

    nav_graph_path: str = field(default_factory=lambda: str(
        Path(str(os.getcwd())).parent.resolve() /
        'DATA/data/disceret/processed/nav_graph_10'),
                                metadata={"help": "nav_graph path"})
    token_dict_path: str = field(default_factory=lambda: str(
        Path(str(os.getcwd())).parent.resolve() /
        'DATA/data/disceret/processed/token_dict_10'),
                                 metadata={"help": "token_dict path"})
    vertices_path: str = field(default_factory=lambda: str(
        Path(str(os.getcwd())).parent.resolve() /
        'DATA/data/disceret/scene_meshes'))
    dagger_mode_load_scene: List[str] = field(default_factory=list)
    dagger_update_size: int = field(default=8000)
    dagger_mode: str = field(
        default="end",
        metadata={"help": 'dagger mode in [end middle nearest]'})
    dagger_p: float = field(default=1.0, metadata={"help": 'dagger p'})

    TF_mode_load_scene: List[str] = field(default_factory=list)

    ablate_instruction: bool = field(default=False)
    ablate_rgb: bool = field(default=False)
    ablate_depth: bool = field(default=False)
    SEQ2SEQ_use_prev_action: bool = field(default=False)
    PROGRESS_MONITOR_use: bool = field(default=False)
    PROGRESS_MONITOR_alpha: float = field(default=1.0)

    EVAL_CKPT_PATH_DIR: Optional[str] = field(default=None)
    EVAL_DATASET: str = field(default="val_unseen")
    EVAL_NUM: int = field(default=-1)
    EVAL_GENERATE_VIDEO: bool = field(default=False)

    rgb_encoder_use_place365: bool = field(default=False)
    tokenizer_use_bert: bool = field(default=True)

    simulator_tool_port: int = field(default=30000,
                                     metadata={"help": "simulator_tool port"})
    DDP_MASTER_PORT: int = field(default=20001,
                                 metadata={"help": "DDP MASTER_PORT"})

    continue_start_from_dagger_it: Optional[int] = field(default=None)
    continue_start_from_checkpoint_path: Optional[str] = field(default=None)

    vlnbert: bool = field(default=False)
    featdropout: float = field(default=0.4)
    action_feature: int = field(default=32)

    eval_save_path: Optional[str] = field(default=None)
    dagger_save_path: Optional[str] = field(default=None)
    activate_maps: Optional[List[str]] = field(default_factory=list)

    gpu_id: int = field(default=3, metadata={"help": "simulator gpus"})
    always_help: bool = field(default=False)
    use_gt: bool = field(default=False)

    dataset_path: Optional[str] = field(default=None)

    use_a_star: bool = field(default=False,
                             metadata={"help": "use a star for navigation"})
    device: int = field(default=0, metadata={"help": "device id for training"})
    llm_checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the LLM checkpoint."})


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=True)
    mm_use_im_start_end: bool = field(default=False)


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="facebook/opt-350m")
    model_base: Optional[str] = field(default=None)
    traj_model_path: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    groundingdino_config: Optional[str] = field(default=None)
    groundingdino_model_path: Optional[str] = field(default=None)


parser = transformers.HfArgumentParser(
    (CommonArguments, ModelArguments, DataArguments))
args, model_args, data_args = parser.parse_args_into_dataclasses()

args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = '{}/workdir/{}/logs/{}_{}.log'.format(
    args.project_prefix, args.run_type, args.collect_type, args.make_dir_time)

# args.run_type = 'collect'
assert args.run_type in ['collect', 'train', 'eval'], 'run_type error'
# args.collect_type = 'TF'
assert args.collect_type in ['TF', 'dagger'], 'collect_type error'

args.machines_info = [
    {
        'MACHINE_IP': '127.0.0.1',
        'SOCKET_PORT': int(args.simulator_tool_port),
        'MAX_SCENE_NUM': 16,
        'open_scenes': [],
    },
]

args.TRAIN_VOCAB = Path(
    args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.TRAINVAL_VOCAB = Path(
    args.project_prefix) / 'DATA/data/aerialvln/train_vocab.txt'
args.vocab_size = 10038

default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
default_config.freeze()
