#!/bin/bash

wandb login <...>

concept=${1}            # dog6
placeholder_token=${2}  # sks
class_prompt=${3}       # dog
concept_property=${4}   # object / live
exp_num=${5}
wandb_exp_name=${6}
with_gram=${7}                # flag to use gram matrix of concept's ca maps in a student loss calculation
values_with_gram=${8}         # flag to use gram matrix of values in a student loss calculation
use_ca=${9}                   # flag to use matching of concept's cross-attn maps in a student loss calculation
use_sa=${10}                  # flag to use matching of self-attn maps in a student loss calculation
use_v_mid=${11}
use_v_all=${12}
# regularization_option=${11}   # flag to use any blocks of values in self-attn layers in a student loss calculation
# use_base_prompt_for_v=${11}   # flag to use base prompt for matching of values
cuda_id=${13}

path=""
export MODEL_NAME="${path}/stable-diffusion-xl-base-1.0"
export VAE_PATH="${path}/sdxl-vae-fp16-fix"
export DATA_DIR="${path}/diffusers/datasets/dataset/${concept}"
export OUTPUT_DIR="${path}/ts_db-lora_sdxl/res_TS/00${exp_num}-res-${concept}_TS_db-lora_sdxl"

accelerate launch --gpu_ids=${cuda_id} train_ts_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --instance_data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_prompt=${class_prompt} \
  --instance_prompt="a photo of ${placeholder_token} ${class_prompt}" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_epochs=200 \
  --num_validation_images=3 \
  --checkpointing_steps=500 \
  --seed=0 \
  --mixed_precision="no" \
  --placeholder_token=${placeholder_token} \
  --concept_property=${concept_property} \
  --exp_name="00${exp_num}-res-${concept}_TS_db-lora_sdxl" \
  --wandb_exp_name="${exp_num}-${concept}_${wandb_exp_name}" \
  --use_cross_attn_maps=${use_ca} \
  --use_self_attn_maps=${use_sa} \
  --use_self_attn_maps_v_mid_block=${use_v_mid} \
  --use_self_attn_maps_v_all_blocks=${use_v_all} \
  --with_gram=$( [ "$with_gram" -eq 1 ] && echo "True" || echo "False" ) \
  --values_with_gram=$( [ "$values_with_gram" -eq 1 ] && echo "True" || echo "False" ) \
  --teacher_stopping_step=1200 \
  --initial_validation_epoch=400 \
  # --${regularization_option} \
  # --validation_prompt="a photo of ${placeholder_token} ${class_prompt} in a bucket" \
  # --gpu_ids=${cuda_id} 
