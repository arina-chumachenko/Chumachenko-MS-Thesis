#!/bin/bash

source deactivate
source activate <...>

OUTPUT_DIR=""

# exp_name=${1}
# checkpoint_idx=${2}
beta=${1}
tau=${2}

( cd .. && python -m nb_utils.evaluate --gpu 0 --base_path "${OUTPUT_DIR}" --beta "${beta}" --tau "${tau}" --checkpoints_idxs 300 --exp_names "" )
