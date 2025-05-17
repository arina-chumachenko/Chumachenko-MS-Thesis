#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from lora import LoRACrossAttnProcessor
from nb_utils import eval_sets_core
from ts_utils import (
    SaveOutput,
    define_target_token_id,
    gram_matrix,
    concept_gram_matrix,
    get_attnention_maps,
    get_value,
    get_concept_attn_map,
    calculate_pairwise_mse_loss,
    get_attention_filter_names,
    get_names_of_ca_q_k_blocks,
    get_names_of_sa_v_mid_block,
    get_names_of_sa_v_all_blocks,
    get_names_return_none,
)

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=model_index_filename, revision=revision
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type


def save_model_card(
    repo_id: str,
    use_dora: bool,
    images=None,
    base_model: str = None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    vae_path=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# {'SDXL' if 'playground' not in base_model else 'Playground'} LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    if "playground" in base_model:
        model_description += """\n
## License

Please adhere to the licensing terms as described [here](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++" if "playground" not in base_model else "playground-v2dot5-community",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora" if not use_dora else "dora",
        "template:sd-lora",
    ]
    if "playground" in base_model:
        tags.extend(["playground", "playground-diffusers"])
    else:
        tags.extend(["stable-diffusion-xl", "stable-diffusion-xl-diffusers"])

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


validation_prompt_set_live = [
    'a photo of {} {} in a bucket',
    'a {} {} with the Eiffel Tower in the background',
    'a {} {} in a purple wizard outfit',
    # 'a {} on a snowy mountaintop, partially buried under fresh, powdery snow', # background
    # 'a {} on an alien planet', # background
    # 'a traditional Chinese painting of a {}', # style
]    

validation_prompt_set_object = [
    'a {} {} on a stone wall in the countryside',
    'a {} {} with the Eiffel Tower in the background',
    'a purple {} {} on the dining table',
]  

def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    torch_dtype,
    validation_name="validation",
):
    validation_set = validation_prompt_set_object if args.concept_property=="object" else validation_prompt_set_live
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images for each of the prompts:"
        f"{[prompt.format(f'{args.placeholder_token}', f'{args.class_prompt}') for prompt in validation_set]}"
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if not args.do_edm_style_training:
        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    # Currently the context determination is a bit hand-wavy. We can improve it in the future if there's a better
    # way to condition it. Reference: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
    if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        images, captions = [], []
        for prompt in validation_set:
            prompt = prompt.format(f'{args.placeholder_token}', f'{args.class_prompt}')
            imgs = pipeline(
                prompt,
                num_images_per_prompt=args.num_validation_images,
                num_inference_steps=25, 
                generator=generator
            ).images
            images += imgs        
            captions += [prompt] * args.num_validation_images

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(validation_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    validation_name: [
                        wandb.Image(images[i].resize((128, 128)), caption=captions[i]) for i in range(len(images))
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=3,
        help="Number of images that should be generated during validation with validation_prompt_set.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=200,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " from validation_prompt_set multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--initial_validation_epoch",
        type=int,
        default=200,
        help=(
            "An epoch number from which validation is started."
        ),
    )
    parser.add_argument(
        "--do_edm_style_training",
        default=False,
        action="store_true",
        help="Flag to conduct training using the EDM formulation as introduced in https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_kohya_format",
        action="store_true",
        help="Flag to additionally generate final state dict in the Kohya format so that it becomes compatible with A111, Comfy, Kohya, etc.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--initial_checkpointing_step",
        type=int,
        default=500,
        help=(
            "A step number from which checkpointing is started."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_scheduler_student",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use for student model. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--optimizer_student",
        type=str,
        default="AdamW",
        help=('The optimizer type to use for student model. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help=(
            "Wether to train a DoRA as proposed in- DoRA: Weight-Decomposed Low-Rank Adaptation https://arxiv.org/abs/2402.09353. "
            "Note: to use DoRA you need to install peft from main, `pip install git+https://github.com/huggingface/peft.git`"
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="A LoRA rank for lora processor.",
    )
    parser.add_argument(
        "--concept_property", 
        type=str, 
        default="object", 
        help="Choose between 'object' and 'live'"
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--with_gram",
        type=lambda x: x.lower() in ("true", "1"),
        default=False,
        help="Calculate gram matrix or not. Accepts 'True'/'1' or 'False'/'0'."
    )
    parser.add_argument(
        "--values_with_gram",
        type=lambda x: x.lower() in ("true", "1"),
        default=False,
        help="Calculate gram matrix or not. Accepts 'True'/'1' or 'False'/'0'."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        required=True,
        help="The name of the experiment.",
    )
    parser.add_argument(
        "--wandb_exp_name",
        type=str,
        default='0-res-concept_TS_db-lora_sdxl',
        help="The name of the experiment that will be displayed in wandb."
    )
    parser.add_argument(
        "--use_cross_attn_maps",
        default=False,
        type=lambda x: x.lower() in ("true", "1"),
        help="Flag to make a hook on q, k, q_lora and k_lora of cross attention maps of unet. Accepts 'True'/'1' or 'False'/'0'."
    )
    parser.add_argument(
        "--use_self_attn_maps",
        default=False,
        type=lambda x: x.lower() in ("true", "1"),
        help="Flag to make a hook on q, k, q_lora and k_lora of self attention maps of unet. Accepts 'True'/'1' or 'False'/'0'."
    )
    parser.add_argument(
        "--use_self_attn_maps_v_mid_block",
        default=False,
        type=lambda x: x.lower() in ("true", "1"),
        help="Flag to make a hook on v and v_lora of self attention maps of unet middle block. Accepts 'True'/'1' or 'False'/'0'."
    )
    parser.add_argument(
        "--use_self_attn_maps_v_all_blocks",
        default=False,
        type=lambda x: x.lower() in ("true", "1"),
        help="Flag to make a hook on v and v_lora of self attention maps of unet all blocks. Accepts 'True'/'1' or 'False'/'0'."
    )
    # parser.add_argument(
    #     "--use_base_prompt_for_values_matching",
    #     default=False,
    #     type=lambda x: x.lower() in ("true", "1"),
    #     help=(
    #         "Flag to use a base prompt (--instance_prompt) for values matching."
    #         "Accepts 'True'/'1' or 'False'/'0'."
    #     )
    # )
    parser.add_argument(
        "--teacher_stopping_step",
        type=int,
        default=10000,
        help=(
            "A step number when we stop a teacher model."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        tokenizer_one,
        tokenizer_two,
        class_data_root=None,
        class_num=None,
        concept_property="object",  # [object, live]
        placeholder_token='*',
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.concept_property = concept_property
        self.placeholder_token = placeholder_token
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))
        
        self.templates = eval_sets_core.live_set if concept_property == "live" else eval_sets_core.object_set

        # image processing to prepare for using SD-XL micro-conditioning
        self.original_sizes = []
        self.crop_top_lefts = []
        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            self.original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            self.crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        original_size = self.original_sizes[index % self.num_instance_images]
        crop_top_left = self.crop_top_lefts[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["original_size"] = original_size
        example["crop_top_left"] = crop_top_left

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        placeholder_token = self.placeholder_token
        various_prompt = random.choice(self.templates).format(placeholder_token)

        example["various_prompt"] = various_prompt

        word_id_1 = define_target_token_id(self.tokenizer_one, placeholder_token, various_prompt)
        word_id_2 = define_target_token_id(self.tokenizer_two, placeholder_token, various_prompt)

        if word_id_1 != word_id_2:
            raise ValueError(f"Word_id_1 and word_id_2 should be the same, but they are different: {word_id_1} and {word_id_2}.")

        example["word_id_1"] = word_id_1

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example


def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    various_prompts = [example["various_prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]
    word_ids_1 = [example["word_id_1"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "various_prompts": various_prompts,
        "word_ids_1": word_ids_1,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch

class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.do_edm_style_training and args.snr_gamma is not None:
        raise ValueError("Min-SNR formulation is not supported when conducting EDM-style training.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    # if not args.use_self_attn_maps_v_mid_block and not args.use_self_attn_maps_v_all_blocks and args.use_base_prompt_for_values_matching:
    #     raise ValueError(
    #         "Flag --use_base_prompt_for_values_matching can be used only with --use_self_attn_maps_v_mid_block or --use_self_attn_maps_v_all_blocks."
    #     )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    scheduler_type = determine_scheduler_type(args.pretrained_model_name_or_path, args.revision)
    if "EDM" in scheduler_type:
        args.do_edm_style_training = True
        noise_scheduler = EDMEulerScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        logger.info("Performing EDM-style training!")
    elif args.do_edm_style_training:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        logger.info("Performing EDM-style training!")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=args.revision, 
        variant=args.variant
    )
    unet_student = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=args.revision, 
        variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    unet_student.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    unet_student.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            unet_student.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        unet_student.enable_gradient_checkpointing()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    lora_attn_processor = LoRACrossAttnProcessor
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[
                block_id
            ]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim:
            rank = min(cross_attention_dim, hidden_size, args.lora_rank)
        else:
            rank = min(hidden_size, args.lora_rank)
        kwargs = {
            "hidden_size": hidden_size,
            "cross_attention_dim": cross_attention_dim,
            "rank": rank,
        }
        lora_attn_procs[name] = lora_attn_processor(**kwargs)

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    lora_attn_procs_student = {}
    for name in unet_student.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet_student.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size_student = unet_student.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size_student = list(reversed(unet_student.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size_student = unet_student.config.block_out_channels[block_id]
        if cross_attention_dim:
            rank = min(cross_attention_dim, hidden_size_student, args.lora_rank)
        else:
            rank = min(hidden_size_student, args.lora_rank)
        kwargs = {
            "hidden_size": hidden_size_student,
            "cross_attention_dim": cross_attention_dim,
            "rank": rank,
        }
        lora_attn_procs_student[name] = lora_attn_processor(**kwargs)

    unet_student.set_attn_processor(lora_attn_procs_student)
    lora_layers_student = AttnProcsLayers(unet_student.attn_processors)

    for name, layer in unet.named_modules():
        layer.module_name = name
    for name, layer in unet_student.named_modules():
        layer.module_name = name

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
    if args.mixed_precision == "fp16":
        models_student = [unet_student]
        cast_training_params(models_student, dtype=torch.float32)

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    unet_student_lora_parameters = list(filter(lambda p: p.requires_grad, unet_student.parameters()))

    # Optimization parameters
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [unet_lora_parameters_with_lr]

    unet_student_lora_parameters_with_lr = {"params": unet_student_lora_parameters, "lr": args.learning_rate}
    params_student_to_optimize = [unet_student_lora_parameters_with_lr]

    # Optimizer creation
    for optimizer_model in [args.optimizer, args.optimizer_student]:
        if not (optimizer_model.lower() == "prodigy" or optimizer_model.lower() == "adamw"):
            logger.warning(
                f"Unsupported choice of optimizer: {optimizer_model}.Supported optimizers include [adamW, prodigy]."
                "Defaulting to adamW"
            )
            optimizer_model = "adamw"

    for optimizer_model in [args.optimizer, args.optimizer_student]:
        if args.use_8bit_adam and not optimizer_model.lower() == "adamw":
            logger.warning(
                f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
                f"set to {optimizer_model.lower()}"
            )

    if args.optimizer.lower() == "adamw" and args.optimizer_student.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
            optimizer_student_class = bnb.optim.AdamW8bit            
        else:
            optimizer_class = torch.optim.AdamW
            optimizer_student_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        optimizer_student = optimizer_student_class(
            params_student_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy" and args.optimizer_student.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy
        optimizer_student_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
        optimizer_student = optimizer_student_class(
            params_student_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        placeholder_token=args.placeholder_token,
        class_prompt=args.class_prompt,
        class_data_root=None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings
    # pooled text embeddings
    # time ids

    def compute_time_ids(original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Clear the memory here
    if not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.
    if not train_dataset.custom_instance_prompts:
        prompt_embeds = instance_prompt_hidden_states
        unet_add_text_embeds = instance_pooled_prompt_embeds

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    lr_scheduler_student = get_scheduler(
        args.lr_scheduler_student,
        optimizer=optimizer_student,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, unet_student, optimizer, optimizer_student, train_dataloader, lr_scheduler, lr_scheduler_student  = accelerator.prepare(
        unet, unet_student, optimizer, optimizer_student, train_dataloader, lr_scheduler, lr_scheduler_student 
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = (
            "dreambooth-lora-sd-xl"
            if "playground" not in args.pretrained_model_name_or_path
            else "dreambooth-lora-playground"
        )
        accelerator.init_trackers(tracker_name, config=vars(args), init_kwargs={"wandb": {"name": args.wandb_exp_name}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        unet_student.train()
        for step, batch in enumerate(train_dataloader):
            # do_detach = (global_step >= args.teacher_stopping_step)

            # with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
            prompts = batch["prompts"]

            # encode batch prompts when custom prompts are provided for each image -
            if train_dataset.custom_instance_prompts:
                prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )

            # Convert images to latent space
            model_input = vae.encode(pixel_values).latent_dist.sample()

            if latents_mean is None and latents_std is None:
                model_input = model_input * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    model_input = model_input.to(weight_dtype)
            else:
                latents_mean = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
                latents_std = latents_std.to(device=model_input.device, dtype=model_input.dtype)
                model_input = (model_input - latents_mean) * vae.config.scaling_factor / latents_std
                model_input = model_input.to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            if not args.do_edm_style_training:
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()
            else:
                # in EDM formulation, the model is conditioned on the pre-conditioned noise levels
                # instead of discrete timesteps, so here we sample indices to get the noise levels
                # from `scheduler.timesteps`
                indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
            # For EDM-style training, we first obtain the sigmas based on the continuous timesteps.
            # We then precondition the final model inputs based on these sigmas instead of the timesteps.
            # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
            if args.do_edm_style_training:
                sigmas = get_sigmas(timesteps, len(noisy_model_input.shape), noisy_model_input.dtype)
                if "EDM" in scheduler_type:
                    inp_noisy_latents = noise_scheduler.precondition_inputs(noisy_model_input, sigmas)
                else:
                    inp_noisy_latents = noisy_model_input / ((sigmas**2 + 1) ** 0.5)

            # time ids
            add_time_ids = torch.cat(
                [
                    compute_time_ids(original_size=s, crops_coords_top_left=c)
                    for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                ]
            )

            # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
            if not train_dataset.custom_instance_prompts:
                elems_to_repeat_text_embeds = bsz
            else:
                elems_to_repeat_text_embeds = 1

            # Predict the noise residual
            unet_added_conditions = {
                "time_ids": add_time_ids,
                "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
            }
            prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)

            # if global_step <= args.teacher_stopping_step:
            model_pred = unet(
                inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                timesteps,
                prompt_embeds_input,
                added_cond_kwargs=unet_added_conditions,
                return_dict=False,
            )[0]

            weighting = None
            if args.do_edm_style_training:
                # Similar to the input preconditioning, the model predictions are also preconditioned
                # on noised model inputs (before preconditioning) and the sigmas.
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                if "EDM" in scheduler_type:
                    model_pred = noise_scheduler.precondition_outputs(noisy_model_input, model_pred, sigmas)
                else:
                    if noise_scheduler.config.prediction_type == "epsilon":
                        model_pred = model_pred * (-sigmas) + noisy_model_input
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                            noisy_model_input / (sigmas**2 + 1)
                        )
                # We are not doing weighting here because it tends result in numerical problems.
                # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                # There might be other alternatives for weighting as well:
                # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                if "EDM" not in scheduler_type:
                    weighting = (sigmas**-2.0).float()

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = model_input if args.do_edm_style_training else noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = (
                    model_input
                    if args.do_edm_style_training
                    else noise_scheduler.get_velocity(model_input, noise, timesteps)
                )
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.snr_gamma is None:
                if weighting is not None:
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                            target.shape[0], -1
                        ),
                        1,
                    )
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                base_weight = (
                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )

                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective needs to be floored to an SNR weight of one.
                    mse_loss_weights = base_weight + 1
                else:
                    # Epsilon and sample both use the same loss weights.
                    mse_loss_weights = base_weight

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (unet_lora_parameters)
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # else:
            #     with torch.no_grad():
            #         model_pred = unet(
            #             inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
            #             timesteps,
            #             prompt_embeds_input,
            #             added_cond_kwargs=unet_added_conditions,
            #             return_dict=False,
            #         )[0]


            # Various prompts
            various_prompts = batch["various_prompts"]
            various_prompt_hidden_states, various_pooled_prompt_embeds = compute_text_embeddings(
                various_prompts, [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two]
            )
            various_prompt_embeds = various_prompt_hidden_states
            unet_add_various_text_embeds = various_pooled_prompt_embeds

            word_ids = batch["word_ids_1"]

            # Predict the noise residual
            unet_various_added_conditions = {
                "time_ids": add_time_ids,
                "text_embeds": unet_add_various_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
            }
            various_prompt_embeds_input = various_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)

            filter_names = None 
            condition = None
            condition_2 = None

            if args.use_cross_attn_maps or args.use_self_attn_maps:
                filter_names = get_attention_filter_names(use_cross_attn=args.use_cross_attn_maps, use_self_attn=args.use_self_attn_maps)
                condition = get_names_of_ca_q_k_blocks
            # if args.use_self_attn_maps:
            #     use_self_attn=True
            #     # filter_names = get_attention_filter_names(use_self_attn=True)
            #     condition = get_names_of_ca_q_k_blocks

            # if args.use_self_attn_maps_v_mid_block:
            #     condition = get_names_of_sa_v_mid_block
            # if args.use_self_attn_maps_v_all_blocks:
            #     condition = get_names_of_sa_v_all_blocks

            if args.use_self_attn_maps_v_mid_block: # and args.use_base_prompt_for_values_matching:
                condition = get_names_of_ca_q_k_blocks if (args.use_cross_attn_maps or args.use_self_attn_maps) else get_names_return_none
                condition_2 = get_names_of_sa_v_mid_block
                # with_ca_maps_2 = False
            if args.use_self_attn_maps_v_all_blocks: # and args.use_base_prompt_for_values_matching:
                condition = get_names_of_ca_q_k_blocks if (args.use_cross_attn_maps or args.use_self_attn_maps) else get_names_return_none
                condition_2 = get_names_of_sa_v_all_blocks
                # with_ca_maps_2 = False
            if not args.use_cross_attn_maps and not args.use_self_attn_maps:
                condition = get_names_return_none
            if not args.use_self_attn_maps_v_mid_block and not args.use_self_attn_maps_v_all_blocks:
                condition_2 = get_names_return_none

            only_mid_block = args.use_self_attn_maps_v_mid_block

            if global_step == 0:
                print(f"\nfilter_names: {filter_names}, \nwith_ca_maps: {args.use_cross_attn_maps}, \nwith_sa_maps: {args.use_self_attn_maps}")
            
            with torch.no_grad():
                # if args.use_base_prompt_for_values_matching:
                # Hook for unet with various prompt 
                save_output = SaveOutput(do_detach=True)
                hook_handles = []
                for name, layer in unet.named_modules():
                    if condition(name, filter_names, with_ca_maps=True):
                        if global_step == 0:
                            print(name)
                        handle = layer.register_forward_hook(save_output)
                        hook_handles.append(handle)

                _ = unet(
                    inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                    timesteps,
                    prompt_embeds_input,
                    added_cond_kwargs=unet_added_conditions,
                    # various_prompt_embeds_input,
                    # added_cond_kwargs=unet_various_added_conditions,
                    return_dict=False,
                )[0]

                if args.use_cross_attn_maps:
                    attention_probs_list = get_attnention_maps(save_output, unet, which_attn='attn2', use_lora=True, do_detach=False)
                    concept_attn_maps = get_concept_attn_map(attention_probs_list, bsz, word_ids)
                    if args.with_gram:
                        print("Used gram!!!")
                        concept_attn_maps = [
                            torch.sqrt(concept_gram_matrix(conc_attn_map)) 
                            for conc_attn_map in concept_attn_maps
                        ]
                    
                if args.use_self_attn_maps:
                    self_attention_probs_list = get_attnention_maps(save_output, unet, which_attn='attn1', use_lora=True, do_detach=False)
                    concept_self_attn_maps = get_concept_attn_map(self_attention_probs_list, bsz, word_ids)

                for handle in hook_handles:
                    handle.remove()
                del save_output
            
                if args.use_self_attn_maps_v_mid_block or args.use_self_attn_maps_v_all_blocks:
                    # Hook for unet with base prompt 
                    save_output = SaveOutput(do_detach=True)
                    hook_handles = []
                    for name, layer in unet.named_modules():
                        if condition_2(name, filter_names, with_ca_maps=False):
                            if global_step == 0:
                                print(name)
                            handle = layer.register_forward_hook(save_output)
                            hook_handles.append(handle)

                    _ = unet(
                        inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]
                    
                    values = get_value(save_output, unet, use_lora=True, only_mid_block=only_mid_block)
                    if args.values_with_gram:
                        values = [gram_matrix(value) for value in values]

                    for handle in hook_handles:
                        handle.remove()
                    del save_output

                # else: # use various prompt to get values from self-attn layers 
                #     # Hook for unet with various prompt 
                #     save_output = SaveOutput(do_detach=True)
                #     hook_handles = []
                #     if global_step == 0:
                #         print('\nTeacher layers:')
                #     for name, layer in unet.named_modules():
                #         if condition(name, filter_names, with_ca_maps):
                #             if global_step == 0:
                #                 print(name)
                #             handle = layer.register_forward_hook(save_output)
                #             hook_handles.append(handle)

                #     _ = unet(
                #         inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                #         timesteps,
                #         prompt_embeds_input,
                #         added_cond_kwargs=unet_added_conditions,
                #         # various_prompt_embeds_input,
                #         # added_cond_kwargs=unet_various_added_conditions,
                #         return_dict=False,
                #     )[0]

                #     if args.use_cross_attn_maps:
                #         attention_probs_list = get_attnention_maps(save_output, unet, which_attn='attn2', use_lora=True, do_detach=do_detach)
                #         concept_attn_maps = get_concept_attn_map(attention_probs_list, bsz, word_ids)
                #         if args.with_gram:
                #             concept_attn_maps = [
                #                 torch.sqrt(concept_gram_matrix(ca_map)) 
                #                 for ca_map in concept_attn_maps
                #             ]

                #     if args.use_self_attn_maps:
                #         self_attention_probs_list = get_attnention_maps(save_output, unet, which_attn='attn1', use_lora=True, do_detach=do_detach)
                #         concept_self_attn_maps = get_concept_attn_map(self_attention_probs_list, bsz, word_ids)
                    
                #     if args.use_self_attn_maps_v_mid_block or args.use_self_attn_maps_v_all_blocks:
                #         values = get_value(save_output, unet, use_lora=True, only_mid_block=only_mid_block)
                #         if args.values_with_gram:
                #             values = [gram_matrix(value) for value in values]

                #     for handle in hook_handles:
                #         handle.remove()
                #     del save_output

            # Student UNet
            # if args.use_base_prompt_for_values_matching:
            # Hook for student unet with various prompt 
            save_output = SaveOutput(do_detach=False)
            hook_handles = []
            for name, layer in unet_student.named_modules():
                if condition(name, filter_names, with_ca_maps=True):
                    if global_step == 0:
                        print(name)
                    handle = layer.register_forward_hook(save_output)
                    hook_handles.append(handle)

            # if args.use_cross_attn_maps or args.use_self_attn_maps:
            _ = unet_student(
                inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                timesteps,
                prompt_embeds_input,
                added_cond_kwargs=unet_added_conditions,
                # various_prompt_embeds_input,
                # added_cond_kwargs=unet_various_added_conditions,
                return_dict=False,
            )[0]
            if args.use_cross_attn_maps:
                student_attention_probs_list = get_attnention_maps(save_output, unet_student, which_attn='attn2', use_lora=True) 
                student_concept_attn_maps = get_concept_attn_map(student_attention_probs_list, bsz, word_ids)
                if args.with_gram:
                    print("\nUsed gram!!!")
                    student_concept_attn_maps = [
                        torch.sqrt(concept_gram_matrix(st_conc_attn_map)) 
                        for st_conc_attn_map in student_concept_attn_maps
                    ]

            if args.use_self_attn_maps:
                student_self_attention_probs_list = get_attnention_maps(save_output, unet_student, which_attn='attn1', use_lora=True) 
                student_concept_self_attn_maps = get_concept_attn_map(student_self_attention_probs_list, bsz, word_ids)

            for handle in hook_handles:
                handle.remove()
            del save_output

            if args.use_self_attn_maps_v_mid_block or args.use_self_attn_maps_v_all_blocks:
                # with torch.no_grad():
                # Hook for student unet with base prompt 
                save_output = SaveOutput(do_detach=False)
                hook_handles = []
                for name, layer in unet_student.named_modules():
                    if condition_2(name, filter_names, with_ca_maps=False):
                        if global_step == 0:
                            print(name)
                        handle = layer.register_forward_hook(save_output)
                        hook_handles.append(handle)

                _ = unet_student(
                    inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                    timesteps,
                    prompt_embeds_input,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                if args.use_self_attn_maps_v_mid_block or args.use_self_attn_maps_v_all_blocks:
                    values_student = get_value(save_output, unet_student, use_lora=True, only_mid_block=only_mid_block)
                    if args.values_with_gram:
                        values_student = [gram_matrix(value) for value in values_student]

                for handle in hook_handles:
                    handle.remove()
                del save_output
                
            # else:
            #     # Hook for student unet with various prompt 
            #     save_output = SaveOutput(do_detach=False)
            #     hook_handles = []
            #     for name, layer in unet_student.named_modules():
            #         if condition(name, filter_names, with_ca_maps):
            #             if global_step == 0:
            #                 print(name)
            #             handle = layer.register_forward_hook(save_output)
            #             hook_handles.append(handle)
                
            #     if args.use_cross_attn_maps or args.use_self_attn_maps:
            #         _ = unet_student(
            #             inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
            #             timesteps,
            #             prompt_embeds_input,
            #             added_cond_kwargs=unet_added_conditions,
            #             # various_prompt_embeds_input,
            #             # added_cond_kwargs=unet_various_added_conditions,
            #             return_dict=False,
            #         )[0]

            #         student_attention_probs_list = get_attnention_maps(save_output, unet_student, which_attn='attn2', use_lora=True) 
            #         student_concept_attn_maps = get_concept_attn_map(student_attention_probs_list, bsz, word_ids)
            #         if args.with_gram:
            #             student_concept_attn_maps = [
            #                 torch.sqrt(concept_gram_matrix(st_conc_attn_map)) 
            #                 for st_conc_attn_map in student_concept_attn_maps
            #             ]

                    
            #         student_self_attention_probs_list = get_attnention_maps(save_output, unet_student, which_attn='attn1', use_lora=True) 
            #         student_concept_self_attn_maps = get_concept_attn_map(student_self_attention_probs_list, bsz, word_ids)

            #     if args.use_self_attn_maps_v_mid_block or args.use_self_attn_maps_v_all_blocks:
            #         values_student = get_value(save_output, unet_student, use_lora=True, only_mid_block=only_mid_block)
            #         if args.values_with_gram:
            #             values_student = [gram_matrix(value) for value in values_student]

            #     for handle in hook_handles:
            #         handle.remove()
            #     del save_output

            loss_student = 0

            if args.use_cross_attn_maps:
                loss_student += calculate_pairwise_mse_loss(concept_attn_maps, student_concept_attn_maps)  
            if args.use_self_attn_maps:
                loss_student += calculate_pairwise_mse_loss(concept_self_attn_maps, student_concept_self_attn_maps)  
            if args.use_self_attn_maps_v_mid_block or args.use_self_attn_maps_v_all_blocks:
                loss_student += torch.sqrt(calculate_pairwise_mse_loss(values, values_student))

            # if not isinstance(loss_student, int) and not isinstance(loss_student, float):
            accelerator.backward(loss_student)
            if accelerator.sync_gradients:
                params_student_to_clip = (unet_student_lora_parameters)
                accelerator.clip_grad_norm_(params_student_to_clip, args.max_grad_norm)

            optimizer_student.step()
            lr_scheduler_student.step()
            optimizer_student.zero_grad()

            # loss_total = loss + loss_student

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    # if (
                    #     global_step >= args.initial_checkpointing_step and 
                    #     global_step <= (args.initial_checkpointing_step+1500) and 
                    #     global_step % args.checkpointing_steps == 0
                    # ):
                    if global_step >= args.initial_checkpointing_step and global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        torch.save(
                            lora_layers.state_dict(),
                            os.path.join(save_path, "pytorch_lora_weights.safetensors")
                        )
                        torch.save(
                            lora_layers_student.state_dict(),
                            os.path.join(save_path, "pytorch_lora_weights_student.safetensors")
                        )              

            logs = {
                "loss_teacher": loss.detach().item(), 
                "loss_student": loss_student.detach().item(), 
                # "loss_total": loss_total.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if global_step >= args.initial_validation_epoch-1 and global_step % args.validation_epochs == 0: 
                print("\nValidation! \n")
                # print(f"epoch: {epoch}")
                # create pipeline
                text_encoder_one = text_encoder_cls_one.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    revision=args.revision,
                    variant=args.variant,
                )
                text_encoder_two = text_encoder_cls_two.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="text_encoder_2",
                    revision=args.revision,
                    variant=args.variant,
                )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    unet=accelerator.unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )

                images = log_validation(
                    pipeline,
                    args,
                    accelerator,
                    epoch,
                    validation_name="validation_teacher",
                    torch_dtype=weight_dtype,
                )
                pipeline_student = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    unet=accelerator.unwrap_model(unet_student),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                images_student = log_validation(
                    pipeline_student,
                    args,
                    accelerator,
                    # pipeline_args,
                    epoch,
                    torch_dtype=weight_dtype,
                    validation_name="validation_student",
                )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub:
            save_model_card(
                repo_id,
                use_dora=args.use_dora,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            save_model_card(
                repo_id,
                use_dora=args.use_dora,
                images=images_student,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
