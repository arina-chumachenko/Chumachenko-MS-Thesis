#!/bin/bash

concept_name=${1}
placeholder_token=${2}
class_name=${3}
checkpoint_idx=${4}
number_of_res_pics=${5}
prompt_subset_length=${6}

path="/home/jovyan/shares/SR006.nfs2/aschumachenko/aschumachenko"
OUTPUT_DIR="${path}/core/res_CR"

CUDA_VISIBLE_DEVICES=0
python vis_results.py \
  --pretrained_model_name="${OUTPUT_DIR}" \
  --checkpoint_idx "${checkpoint_idx}" \
  --concept_name "${concept_name}" \
  --placeholder_token "${placeholder_token}" \
  --class_name "${class_name}" \
  --number_of_res_pics "${number_of_res_pics}" \
  --prompt_subset_length "${prompt_subset_length}" \
  