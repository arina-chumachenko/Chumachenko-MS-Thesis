!/bin/bash

exp_name=${1} 
checkpoint_idx=${2}

path=""
OUTPUT_DIR="${path}/CoRe/res_CR"

python inference.py \
  --pretrained_model_name="${OUTPUT_DIR}/${exp_name}" \
  --config_path "${OUTPUT_DIR}/${exp_name}/logs/hparams.yml" \
  --checkpoint_idx "${checkpoint_idx}" \
  
