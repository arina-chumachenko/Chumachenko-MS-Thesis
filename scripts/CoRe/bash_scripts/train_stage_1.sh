#!/bin/bash

wandb login 

concept=${1}           
placeholder_token=${2} # <${concept}> / sks
initializer_token=${3}
superclass=${4}
concept_property=${5}  # object / live
emb_reg=${6}
attn_reg=${7}
exp_num=${8}
wandb_exp_name=${9}
with_gram=${10}
cuda_id=${11}

path=""
export MODEL_NAME="${path}/stable-diffusion-xl-base-1.0"
export DATA_DIR="${path}/datasets/dataset/${concept}"
export OUTPUT_DIR="${path}/CoRe/res_CR/00${exp_num}-res-${concept}_CR_sdxl"

accelerate launch --gpu_ids=${cuda_id} train_stage_1.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --exp_name="00${exp_num}-res-${concept}_CR_sdxl" \
  --placeholder_token="${placeholder_token}" \
  --initializer_token="${initializer_token}" \
  --class_name="${superclass}" \
  --concept_property="${concept_property}" \
  --train_batch_size=5 \
  --resolution=512 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=300 \
  --learning_rate=5e-3 \
  --num_validation_images=3 \
  --scale_lr \
  --seed=0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --save_steps=300 \
  --validation_steps=100 \
  --checkpointing_steps=100 \
  --wandb_exp_name="${exp_num}-${concept}_${wandb_exp_name}" \
  --add_emb_reg=$( [ "$emb_reg" -eq 1 ] && echo "True" || echo "False" ) \
  --add_attn_reg=$( [ "$attn_reg" -eq 1 ] && echo "True" || echo "False" ) \
  --lambda_emb=1.5e-4 \
  --lambda_attn=0.05 \
  --with_gram=${with_gram} \
  
