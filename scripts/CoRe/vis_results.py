import os
import argparse
import random
import torch
import torch.backends.cuda
import matplotlib.pyplot as plt 
from PIL import Image
import textwrap

import sys
sys.path.append('./core/')
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
    
    exp_nums = ['']

    method_name = 'CR_sdxl'
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
    
        for i, exp_num in enumerate(exp_nums):
            for j, prompt in enumerate(prompt_subset):
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
