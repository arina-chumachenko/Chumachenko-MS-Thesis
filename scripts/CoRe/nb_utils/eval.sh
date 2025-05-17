#!/bin/bash
#SBATCH --job-name=eval                     # Название эксперимента
#SBATCH --error=/home/aalanov/vsoboleva/vs/runs/eval%j.err             # Файл для вывода ошибок 
#SBATCH --output=/home/aalanov/vsoboleva/vs/runs/eval%j.log            # Файл для вывода результатов
#SBATCH --gpus=1                             # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=12                   # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=30:00:00                       # Максимальное время выполнения, после его окончания програмаа просто сбрасывается
#SBATCH -A proj_1430


source deactivate
source activate vsdiffusion

OUTPUT_DIR="/home/aalanov/vsoboleva/vs/res/"

# exp_name=${1}
# checkpoint_idx=${2}
beta=${1}
tau=${2}

( cd .. && python -m nb_utils.evaluate --gpu 0 --base_path "${OUTPUT_DIR}" --beta "${beta}" --tau "${tau}" --checkpoints_idxs 25 200 --exp_names 00033-e7b3-dog6 )
