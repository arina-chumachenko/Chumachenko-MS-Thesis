import os
import argparse
import random
import torch
import torch.backends.cuda
import matplotlib.pyplot as plt 
from PIL import Image
import textwrap

import sys
sys.path.append('/home/jovyan/shares/SR006.nfs2/aschumachenko/aschumachenko/core/')
from nb_utils.eval_sets import base_set, live_set_core, object_set_core
from nb_utils.configs import live_object_data

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")

    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Do not perform actual inference. Only show what prompts will be used for inference"
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        required=True,
        help="Path to pretrained model"
    )    
    parser.add_argument(
        "--checkpoint_idx",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0
    )
    parser.add_argument(
        "--number_of_res_pics",
        type=int,
        default=5
    )
    parser.add_argument(
        "--inference_pictures_num",
        type=int,
        default=5,
        help="A default number of pictures for each prompt after inference."
    )
    parser.add_argument(
        "--prompt_subset_length",
        type=int,
        default=9,
        help="The length of prompt subset without base prompt."
    )
    # parser.add_argument(
    #     "--number_of_exps",
    #     type=int,
    #     default=5,
    #     help="The number of experiments to visualise."
    # )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help="The name of the prompt class"
    )
    parser.add_argument(
        "--concept_name",
        type=str,
        default=None,
        help="The name of the concept"
    )
    return parser.parse_args()


def split_prompt(prompt, max_width=25, num_lines=3):
    """
    Разбивает строку `prompt` на заданное количество строк.
    """
    words = prompt.split()
    wrapped = textwrap.wrap(" ".join(words), width=max_width)
    while len(wrapped) < num_lines:
        wrapped.append("")
    return "\n".join(wrapped[:num_lines])


def main(args):
    
    if 'V100' in torch.cuda.get_device_name(torch.device('cuda')):
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        
    if live_object_data[args.class_name] == 'live':
        prompt_set = live_set_core
    else:
        prompt_set = object_set_core
    
    # exp_nums = [f"{num:03}" for num in range(1, args.number_of_exps+1)]
    exp_nums = ['000', '001', '002', '003', '004', '005', '006'] # , '007', '008'

    method_name = 'CR_sdxl' # 'CR'/'CD'
    folder_name = '3'
    
    labels = [
        'ti_emb_attn + db',             # 0
        'ti + db',                      # 1
        'ti_emb + db',                  # 2
        'ti_attn + db',                 # 3
        'ti_emb_gram + db',             # 4
        'ti_emb_attn + db_attn_noft',   # 5
        'ti_emb_attn + db_gram_noft',   # 6
        # 'ti',                         # 7
        # 'db_lora',                    # 8
    ]

    save_path = os.path.join(args.pretrained_model_name, 'all_results', f'{args.concept_name}', f'{method_name}_chp{args.checkpoint_idx}', f'{folder_name}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for picture_num in range(args.number_of_res_pics):  
        image_idx = random.randint(0, args.inference_pictures_num-1)
        prompt_subset = random.sample(prompt_set, args.prompt_subset_length)
        prompt_subset = base_set + prompt_subset

        _, axs = plt.subplots(
            len(exp_nums), len(prompt_subset), 
            figsize=(len(prompt_subset) * 2 - 1.5, len(exp_nums) * 2 - 0.55),
            gridspec_kw={'wspace': 0, 'hspace': 0} 
        )
    
        # for ax in axs.flat:
        #     ax.axis('off')
    
        for i, exp_num in enumerate(exp_nums):
            for j, prompt in enumerate(prompt_subset):
                # if i == 0 or i == 1:
                #     image_path = os.path.join(
                #     args.pretrained_model_name, f'00{i}-res-{args.concept_name}_CR/checkpoint-{args.checkpoint_idx}/',
                #     'samples', f'ns{args.num_inference_steps}_gs{args.guidance_scale}',
                #     'version_0', prompt.format(f"{args.placeholder_token}"), f'{image_idx}.png'
                # )
                # else:
                image_path = os.path.join(
                    args.pretrained_model_name, f'{exp_num}-res-{args.concept_name}_{method_name}/checkpoint-{args.checkpoint_idx}/',
                    'samples', f'ns{args.num_inference_steps}_gs{args.guidance_scale}',
                    'version_0', prompt.format(f"{args.placeholder_token}"), f'{image_idx}.png'
                )
                image = Image.open(image_path)
                axs[i, j].set_ylabel(labels[i], rotation=90, size='small', va='center', ha='right')
                axs[i, j].imshow(image)
                axs[i, j].axis('off')
    
        for ax, prompt in zip(axs[0, :], prompt_subset):
            ax.set_title(split_prompt(prompt), size='small', pad=5)

        # for ax, label in zip(axs[0], labels):
        #     ax.set_ylabel(f'{label}', rotation=90, size='small', va='center', ha='right', labelpad=5) # labelpad=5,
    
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{picture_num}.png'))
        
    path_to_labels = os.path.join(save_path, 'methods.txt')
    with open(path_to_labels, 'w') as f:
        for label in labels:
            f.write(label + '\n')
    
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)


# 'ti+db_gram_noft_1e5 (sqrt pairwise loss)', # 24
# 'ti+db_attn_noft_1e3 (pairwise_loss)', # 26
# 'ti_emb_1.0+db', 'ti_emb_10.0+db', 'ti_emb_1e-1+db', 'ti_emb_1e3+db', 'ti_emb_1.5e-4+db', 'ti_emb_1e8+db', 'ti_emb_1e-3+db', 'ti_emb_1e-2+db'
# 'ti+db_gram_noft_1e5 (sqrt(gram), interp to max)', 'ti+db_gram_noft_1e5 (sqrt(gram), pairwise_loss)', 'ti+db_attn_noft_1e3 (interp to max)', 'ti+db_attn_noft_1e3 (pairwise_loss)', 'ti+db_attn_noft_1e3 (interp to min)', 'ti+db_gram_noft_1e11 (interp to min)', 
# 'ti+db_attn_1e2',
# 'ti+db_gram_10',
# 'ti_emb_attn+db', 
# 'ti_emb_attn+db_attn', 
# 'ti_emb_attn+db_attn_noft',
# 'ti_emb_attn+db_gram_noft',
# 'ti+db_attn_noft_1e3',
# 'ti+db_gram_noft_1e11',
# 'ti+db_gram_noft_1e10',
# 'ti+db_gram_noft_1e12',
# 'ti_emb_1.0_db',
# 'ti_emb_10.0_db',
# 'ti_emb_0.1_db',
# 'ti_emb_1e3_db'
# 'cd+db', # 1
# 'cd_emb+db'
# 'cd_attn_noft+db'
# 'cd_emb_attn+db'
# 'cd_gram+db'
# 'cd_attn_noft_1e3+db', 
# 'cd_gram_noft_1e11+db', 
# 'cd_gram_noft_1e5+db (sqrt(gram), interp to max)',
# 'cd_gram_noft_1e5+db (sqrt pairwise loss)', # 14
# 'cd+db_gram_noft_1e5 (sqrt pairwise loss)', # 9,
# 'cd_attn_noft_1e3+db (interp to max)',
# 'cd_attn_noft_1e3+db (pairwise_loss)',
# 'cd+db_attn_noft_1e3',
# 'cd+db_gram_noft_1e11',