import argparse
import math
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import shutil
import time
from functools import partial

import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataset.airsim_dataset import AirSimDataset, collate_fn
from models.image_processing_qwen2_vl import Qwen2VLImageProcessor
from models.pilot_llm import PilotLLM
from models.processing_qwen2_vl import Qwen2VLProcessor
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import SchedulerType, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.45.0.dev0")

logger = get_logger(__name__)

require_version(
    "datasets>=2.0.0",
    "To fix: pip install -r examples/pytorch/semantic-segmentation/requirements.txt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a image semantic segmentation task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to a pretrained model or model identifier from huggingface.co/models.",
        default="pilot_llm/weights/Qwen2-VL-2B-Instruct",
    )
    parser.add_argument("--dataset_name_or_path",
                        type=list,
                        help="Name of the dataset on the hub.",
                        default=["data/train_data.json"])
    parser.add_argument("--reload_model_path",
                        help="Name of the dataset on the hub.",
                        default=None)
    parser.add_argument("--num_token", type=int, default=784)
    parser.add_argument("--mission", default="prob")
    parser.add_argument("--output_dir",
                        type=str,
                        default="model_output",
                        help="Where to store the final model.")
    parser.add_argument(
        "--lora",
        type=bool,
        help="if use lora to train",
        default=True,
    )
    parser.add_argument("--lora_target_modules",
                        type=list,
                        default=[
                            "q_proj", "o_proj", "k_proj", "v_proj",
                            "gate_proj", "up_proj", "down_proj"
                        ])
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=5,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        help="",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help=
        "Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=1000,
        help=
        "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--checkpoints_total_limit",
                        type=int,
                        default=3,
                        help="")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.output_dir is None:
        raise ValueError("Need an `output_dir` to create a out dir")

    # Deprecation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_semantic_segmentation_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # in the environment
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        #   kwargs_handlers=[kwargs]
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # We set device_specific to True as we want different data augmentation per device.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_dir = os.path.join(args.output_dir, current_time)
    tb_writer = SummaryWriter(log_dir=log_dir)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load pretrained model and image processor

    # image_processor = Qwen2VLImageProcessor.from_pretrained(args.model_name_or_path)
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_name_or_path,
        # image_processor=image_processor,
        padding_side="right")
    model = PilotLLM.from_pretrained(args.model_name_or_path)

    # load pretrain model（seg and depth pretraining）
    if args.reload_model_path is not None:
        model = PeftModel.from_pretrained(model, args.reload_model_path)
        weights = torch.load(
            os.path.join(args.reload_model_path,
                         "pytorch_model/mp_rank_00_model_states.pt"))
        model.load_state_dict(weights['module'], strict=False)

        # If loading a trained main model, do not rewrite a set of lora
        model = model.merge_and_unload()
        if accelerator.is_main_process:
            del weights

    if args.lora:
        lora_config = LoraConfig(
            r=8,
            target_modules=args.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    for n, p in model.named_parameters():
        if any([
                x in n for x in [
                    "pooling_head", "depth_head", "seg_head", "seg_query",
                    "coord_head", "coord_query"
                ]
        ]):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    model.print_trainable_parameters()
    model = model.to(accelerator.device, dtype=weight_dtype)

    train_dataset = AirSimDataset(args.dataset_name_or_path,
                                  video_frame_num=5,
                                  target_interval=30)  # TODO
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=partial(collate_fn,
                           processor=processor,
                           mission=args.mission),
    )

    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_epsilon,
    )

    # check para
    if accelerator.is_main_process:
        rec_txt1 = open(os.path.join(args.output_dir, 'rec_para.txt'), 'w')
        rec_txt2 = open(os.path.join(args.output_dir, 'rec_para_train.txt'),
                        'w')
        for name, para in model.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    checkpointing_steps = int(checkpointing_steps)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps /
                                      num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume:
        if args.resume is not None or args.resume != "latest":
            checkpoint_path = args.resume
            path = os.path.basename(args.resume)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        # manually load model and optimizer states for DeepSpeed
        if hasattr(model, "load_checkpoint"):
            print("!!!!deepspeed engine !!!")
            model.load_checkpoint(checkpoint_path,
                                  load_optimizer_states=True,
                                  load_lr_scheduler_states=True)

        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "checkpoint" in training_difference:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace(
                "checkpoint-", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if args.resume and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                torch.cuda.empty_cache()
                inputs, kwargs = batch
                inputs = inputs.to(accelerator.device, weight_dtype)
                # delta_coord_gt = delta_coord_gt.to(accelerator.device, weight_dtype)
                preds, loss = model(
                    **inputs,
                    **kwargs,
                )

                # We keep track of the loss at each epoch
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_device_train_batch_size)).mean()
                total_loss += avg_loss.item(
                ) / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                save_path = os.path.join(args.output_dir,
                                         f"checkpoint-{completed_steps}")
                accelerator.save_state(save_path)
                unwrapped_model = accelerator.unwrap_model(model)
                # unwrapped_model = unwrapped_model.merge_and_unload(unwrapped_model)
                unwrapped_model.save_pretrained(save_path)
                logger.info(f"Saved state to {save_path}")

                if args.checkpoints_total_limit is not None and accelerator.is_main_process:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [
                        d for d in checkpoints
                        if (d.startswith("checkpoint") and "best" not in d)
                    ]
                    checkpoints = sorted(checkpoints,
                                         key=lambda x: int(x.split("-")[1]))
                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(
                            checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(
                            f"removing checkpoints: {', '.join(removing_checkpoints)}"
                        )

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(
                                args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

            if completed_steps >= args.max_train_steps:
                break

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            _, kwargs = batch
            tb_writer.add_scalar('Loss', loss.detach().item(), completed_steps)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save)


if __name__ == "__main__":
    main()
