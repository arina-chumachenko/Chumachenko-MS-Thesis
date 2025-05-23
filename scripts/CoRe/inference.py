import os
import tqdm
import yaml
import argparse
import torch
import torch.backends.cuda

from diffusers import StableDiffusionXLPipeline, DDIMScheduler, UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTextModelWithProjection #, CLIPTokenizer

import sys
# sys.path.append('path/CoRe/')
from nb_utils.eval_sets import base_set, live_set_core, object_set_core
from nb_utils.configs import live_object_data
from seed import fix_seed 


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
        "--config_path",
        type=str,
        required=True,
        help="Path to hparams.yml"
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
        "--num_images_per_base_prompt",
        type=int,
        default=25,
        help="Number of generated images for each prompt",
    )
    parser.add_argument(
        "--num_images_per_medium_prompt",
        type=int,
        default=5,
        help="Number of generated images for each prompt",
    )
    parser.add_argument(
        "--batch_size_medium",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch_size_base",
        type=int,
        default=5,
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
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--hub_token",
        type=str, 
        default=None, 
        help="The token to use to push to the Model Hub."
    )
    return parser.parse_args()


def main(args):
    if 'V100' in torch.cuda.get_device_name(torch.device('cuda')):
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    with open(args.config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if live_object_data[config['class_name']] == 'live':
        prompt_set = live_set_core
    else:
        prompt_set = object_set_core
    exp_path = config['output_dir']
    exp_name = config.get('exp_name', os.path.split(config['output_dir'])[-1])

    if args.checkpoint_idx is None:
        checkpoint_path = exp_path
    else:
        checkpoint_path = os.path.join(exp_path, f'checkpoint-{args.checkpoint_idx}')
        exp_name = '{0}_pt{1}'.format(exp_name, args.checkpoint_idx)

    scheduler = DDIMScheduler.from_pretrained(
        config['pretrained_model_name_or_path'], subfolder="scheduler"
    )
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     config['pretrained_model_name_or_path'], subfolder="tokenizer"
    # )
    # tokenizer_2 = CLIPTokenizer.from_pretrained(
    #     config['pretrained_model_name_or_path'], subfolder="tokenizer_2"
    # )

    ### For the second stage uncomment this part 
    # unet = UNet2DConditionModel.from_pretrained(
    #     config['pretrained_model_name_or_path'], subfolder="unet"
    # )
    # unet.load_state_dict(torch.load(
    #     os.path.join(checkpoint_path, 'unet.bin')
    # ))
    # text_encoder = CLIPTextModel.from_pretrained(
    #     config['pretrained_model_name_or_path'],
    #     subfolder="text_encoder", 
    #     revision=config['revision']
    # )
    # text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    #     config['pretrained_model_name_or_path'],
    #     subfolder="text_encoder_2", 
    #     revision=config['revision']
    # )
    # # if config['train_text_encoder']:
    # #     text_encoder.load_state_dict(torch.load(
    # #         os.path.join(checkpoint_path, 'text_encoder_1.bin')
    # #     ))
    # #     text_encoder_2.load_state_dict(torch.load(
    # #         os.path.join(checkpoint_path, 'text_encoder_2.bin')
    # #     ))
    ### For the second stage uncomment this part 
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config['pretrained_model_name_or_path'], 
        ### For the second stage uncomment this part 
        # unet=unet,
        # text_encoder=text_encoder,
        # text_encoder_2=text_encoder_2,
        ### For the second stage uncomment this part 
        scheduler=scheduler,
        torch_dtype=torch.float32
    ).to("cuda")
    pipe.safety_checker = None

    pipe.load_textual_inversion(
        args.pretrained_model_name, 
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        weight_name="learned_embeds.safetensors",
    )
    pipe.load_textual_inversion(
        args.pretrained_model_name, 
        text_encoder = pipe.text_encoder_2,
        tokenizer=pipe.tokenizer_2,
        weight_name = "learned_embeds_2.safetensors", 
    )
    pipe = pipe.to("cuda")

    for prompt in tqdm.tqdm(prompt_set):
        samples_path = os.path.join(
            checkpoint_path, 'samples',
            f'ns{args.num_inference_steps}_gs{args.guidance_scale}',
            'version_0', prompt.format(f"{config['placeholder_token']}") # {config['class_name']}")
        )
        if os.path.exists(samples_path) and len(os.listdir(samples_path)) == args.num_images_per_medium_prompt:
            continue

        batch_size = args.batch_size_medium
        generator = fix_seed(prompt, 0, args.seed, 'cuda')
        # shape = (args.num_images_per_medium_prompt, pipe.unet.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size)
        # latents = randn_tensor(shape, generator=generator, dtype=pipe.unet.dtype, device=pipe.unet.device)
        n_batches = (args.num_images_per_medium_prompt - 1) // batch_size + 1
        images = []
        for i in range(n_batches):
            images_batch = pipe(
                prompt.format(f"{config['placeholder_token']}"), # {config['class_name']}"),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, 
                num_images_per_prompt=batch_size,
                generator=generator,
                # latents=latents[i * batch_size: (i + 1) * batch_size],
            ).images
            images += images_batch

        os.makedirs(samples_path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(samples_path, f'{idx}.png'))

    for prompt in tqdm.tqdm(base_set):
        samples_path = os.path.join(
            checkpoint_path, 'samples',
            f'ns{args.num_inference_steps}_gs{args.guidance_scale}',
            'version_0', prompt.format(f"{config['placeholder_token']}") # {config['class_name']}")
        )

        batch_size = args.batch_size_base
        
        generator = fix_seed(prompt, 0, args.seed, 'cuda')
        # shape = (args.num_images_per_base_prompt, pipe.unet.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size)
        # latents = randn_tensor(shape, generator=generator, dtype=pipe.unet.dtype, device=pipe.unet.device)
        n_batches = (args.num_images_per_base_prompt - 1) // batch_size + 1
        images = []
        for i in range(n_batches):
            images_batch = pipe(
                prompt.format(f"{config['placeholder_token']}"), # {config['class_name']}"),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, num_images_per_prompt=batch_size,
                generator=generator,
                # latents=latents[i * batch_size: (i + 1) * batch_size],
            ).images
            images += images_batch

        os.makedirs(samples_path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(samples_path, f'{idx}.png'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
