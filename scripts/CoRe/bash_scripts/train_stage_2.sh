#!/bin/bash

wandb login ""

concept=${1}
placeholder_token=${2} # <${concept}>, sks
initializer_token=${3}
superclass=${4}
concept_property=${5}  # object / live
exp_num=${6}
wandb_exp_name=${7}
with_gram=${8}
add_attn_reg=${9}
cuda_id=${10}

path=""
export MODEL_NAME="${path}/stable-diffusion-xl-base-1.0"
export DATA_DIR="${path}/datasets/dataset/${concept}"
export OUTPUT_DIR="${path}/CoRe/res_CR/00${exp_num}-res-${concept}_CR_sdxl"

accelerate launch --gpu_ids=${cuda_id} train_stage_2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --exp_name="00${exp_num}-res-${concept}_CR_sdxl" \
  --placeholder_token="${placeholder_token}" \
  --initializer_token="${initializer_token}" \
  --class_name="${superclass}" \
  --concept_property="${concept_property}" \
  --train_batch_size=2 \
  --resolution=512 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=300 \
  --learning_rate=2e-6 \
  --num_validation_images=3 \
  --scale_lr \
  --seed=0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --validation_steps=300 \
  --checkpointing_steps=300 \
  --add_attn_reg=${add_attn_reg} \
  --use_noft_unet_for_cls=True \
  --with_gram=${with_gram} \
  --wandb_exp_name="${exp_num}-${concept}_${wandb_exp_name}" \
