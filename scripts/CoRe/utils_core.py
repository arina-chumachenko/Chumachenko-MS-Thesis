import os
import math
import glob
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention as attn
from nb_utils.configs import _LOAD_IMAGE_BACKEND


validation_prompt_set = [
    'a {} on a snowy mountaintop, partially buried under fresh, powdery snow', # background
    'a {} on an alien planet', # background
    'a {} captured in the soft light of an impressionist painting', # style
    'a traditional Chinese painting of a {}', # style
    'a black {} seen from the top', # color change
]

class SaveOutput:
    def __init__(self, register_outputs=True, register_inputs=False):
        self.outputs = {}
        self.inputs = {}
        self.counter = {}
        self.register_outputs = register_outputs
        self.register_inputs = register_inputs

    def __call__(self, module, module_in, module_out):
        if not hasattr(module, 'module_name'):
            raise AttributeError('All modules should have name attr')
        if self.register_outputs:
            self.outputs[module.module_name] = module_out
        if self.register_inputs:
            self.inputs[module.module_name] = module_in

    def clear(self):
        self.outputs = {}
        self.inputs = {}
        self.counter = 0

def define_target_token_id(tokenizer, token, prompt):
    [target_token] = tokenizer.encode(token, add_special_tokens=False)
    prompt_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt'
    ).input_ids.view(-1)
    word_id = torch.argwhere(torch.eq(prompt_ids, target_token)).item()
    return word_id

def gram_matrix(attn_map):
    a, b, c, d = attn_map.size() 
    # a = batch size(=1)
    # b = number of feature maps
    # (c, d) = dimensions of a f. map (N=c*d)
    features = attn_map.reshape(a * b, c * d)  # resize F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def get_cross_attnention_maps(save_output, model, use_lora=False):
    attention_probs_list = []
    for name, layer in model.named_modules():
        if name.endswith('attn2'):
            heads = layer.heads
            name_q = name + '.to_q'
            name_k = name + '.to_k'
            if use_lora:
                name_q_lora = name + '.processor.to_q_lora'
                name_k_lora = name + '.processor.to_k_lora'
                query = save_output.outputs[name_q] + save_output.outputs[name_q_lora]
                key = save_output.outputs[name_k] + save_output.outputs[name_k_lora]
            else: 
                query = save_output.outputs[name_q]
                key = save_output.outputs[name_k]
            query = attn.head_to_batch_dim(layer, query)
            key = attn.head_to_batch_dim(layer, key)

            # get cross attention maps
            attention_probs = attn.get_attention_scores(layer, query, key)

            # mean across heads
            batch_size = attention_probs.shape[0] // heads
            seq_len = attention_probs.shape[1]
            dim = attention_probs.shape[-1]
    
            attention_probs = attention_probs.view(batch_size, heads, seq_len, dim)
            attention_probs = attention_probs.mean(dim=1)

            sqrt_seq_len = int(math.sqrt(seq_len))
            attention_probs = attention_probs.view(batch_size, sqrt_seq_len, sqrt_seq_len, dim) 
            attention_probs_list.append(attention_probs)
    return attention_probs_list

def interpolate_cross_attention_maps(attention_probs_list, with_gram=False, min_seq=False):
    # The interpolation to the smallest dimention of the cross-attention maps
    if min_seq:
        seq_len = min([ca.shape[1] for ca in attention_probs_list])
    else:            
        seq_len = max([ca.shape[1] for ca in attention_probs_list])        
    target_size = (seq_len, seq_len)
    interp_attention_maps = []
    for ca_map in attention_probs_list:
        interp_ca_map = F.interpolate(ca_map.permute(0, 3, 1, 2), size=target_size, mode='bilinear')
        # If use gram regularization 
        if with_gram:
            interp_ca_map = gram_matrix(interp_ca_map)
        else:
            interp_ca_map = interp_ca_map.permute(0, 2, 3, 1)            
        interp_attention_maps.append(interp_ca_map)
    interpolated_attention_maps = torch.stack(interp_attention_maps)
    return interpolated_attention_maps

def evaluate_validation(args, evaluator, images, prompt):
    paths = sorted(glob.glob(os.path.join(args.train_data_dir, '*')))
    train_images = [_LOAD_IMAGE_BACKEND(path) for path in paths]
    train_images_features, _ = evaluator._get_image_features(train_images, args.resolution)
    images_features, _ = evaluator._get_image_features(images, args.resolution)

    image_similarities, _ = evaluator._calc_similarity(train_images_features, images_features)
    clean_label = (
        prompt
        .replace('{0} {1}'.format(args.placeholder_token, args.class_name), '{0}')
        .replace('{0}'.format(args.placeholder_token), '{0}')
    )
    empty_label = clean_label.replace('{0} ', '').replace(' {0}', '')
    empty_label_features = evaluator.evaluator.get_text_features(empty_label)
    text_similarities, _ = evaluator._calc_similarity(empty_label_features, images_features)
    return image_similarities, text_similarities

def calculate_pairwise_mse_loss(attn_maps_1, attn_maps_2, with_sqrt_gram=False, reduction='sum'):
    mse_losses = []
    for map1, map2 in zip(attn_maps_1, attn_maps_2):
        # check that map1 and map2 have same shape (batch_size, seq_len, seq_len, feature_dim)
        if map1.shape != map2.shape:
            raise ValueError("Mismatch in attention map shapes: {} vs {}".format(map1.shape, map2.shape))
        if with_sqrt_gram:
            map1 = gram_matrix(map1.permute(0, 3, 1, 2))
            map2 = gram_matrix(map2.permute(0, 3, 1, 2))
            mse_loss = torch.sqrt(F.mse_loss(map1, map2, reduction=reduction)) # torch.sqrt()
        else:
            mse_loss = F.mse_loss(map1, map2, reduction=reduction) / (map1.shape[-1] - 1)
        mse_losses.append(mse_loss)
    mean_mse_loss = torch.mean(torch.stack(mse_losses))
    return mean_mse_loss
  
