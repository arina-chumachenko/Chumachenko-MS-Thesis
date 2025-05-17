import math
import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention as attn

class SaveOutput:
    def __init__(self, register_outputs=True, register_inputs=False, do_detach=True):
        self.outputs = {}
        self.inputs = {}
        self.counter = {}
        self.register_outputs = register_outputs
        self.register_inputs = register_inputs
        self.do_detach = do_detach

    def __call__(self, module, module_in, module_out):
        if not hasattr(module, 'module_name'):
            raise AttributeError('All modules should have name attr')
        if self.do_detach:
            if not isinstance(module_in, tuple):
                module_in = module_in.detach()
            if not isinstance(module_out, tuple):
                module_out = module_out.detach()
        if self.register_outputs:
            self.outputs[module.module_name] = module_out
        if self.register_inputs:
            self.inputs[module.module_name] = module_in

    def clear(self):
        # del self.outputs, self.inputs, self.counter
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


def get_attention_filter_names(use_self_attn=False, use_cross_attn=False):
    self_attn_layers = [
        'attn1.processor.to_q_lora',
        'attn1.processor.to_k_lora',
        'attn1.to_q',
        'attn1.to_k'
    ]
    cross_attn_layers = [
        'attn2.processor.to_q_lora',
        'attn2.processor.to_k_lora',
        'attn2.to_q',
        'attn2.to_k'
    ]
    
    filter_names = []
    
    if use_self_attn:
        filter_names.extend(self_attn_layers)
    if use_cross_attn:
        filter_names.extend(cross_attn_layers)
        
    if not filter_names:
        filter_names = None
        
    return filter_names


def get_attnention_maps(save_output, model, which_attn='attn2', use_lora=False, do_detach=False):
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

            # get cross(self) attention maps
            attention_probs = attn.get_attention_scores(layer, query, key)

            # mean across heads
            batch_size = attention_probs.shape[0] // heads
            seq_len = attention_probs.shape[1]
            dim = attention_probs.shape[-1]
    
            attention_probs = attention_probs.view(batch_size, heads, seq_len, dim)
            attention_probs = attention_probs.mean(dim=1)

            sqrt_seq_len = int(math.sqrt(seq_len))
            attention_probs = attention_probs.view(batch_size, sqrt_seq_len, sqrt_seq_len, dim) 
            # if do_detach:
            #     attention_probs = attention_probs.detach()
            attention_probs_list.append(attention_probs)
    return attention_probs_list

def get_value(save_output, model, use_lora=False, only_mid_block=False):
    value_list = []

    def check_condition(name):
        if only_mid_block:
            return 'mid_block' in name
        return True 

    for name, layer in model.named_modules():
        if check_condition(name) and name.endswith('attn1'):
            heads = layer.heads
            name_v = name + '.to_v'
            value = save_output.outputs[name_v]
            if use_lora:
                name_v_lora = name + '.processor.to_v_lora'
                value += save_output.outputs[name_v_lora]
            value = attn.head_to_batch_dim(layer, value)
            bsz, seq_len, dim = value.shape
            bsz = bsz // heads
            value = value.view(bsz, heads, seq_len, dim).mean(dim=1)
            sqrt_seq_len = int(math.sqrt(seq_len))
            value = value.view(bsz, sqrt_seq_len, sqrt_seq_len, dim)     
            value_list.append(value)
    return value_list


def gram_matrix(attn_map):
    a, b, c, d = attn_map.size()  
    features = attn_map.view(a * b, c * d) 
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def concept_gram_matrix(concept_attn_map):
    a, b = concept_attn_map.size()  
    G = torch.mm(concept_attn_map, concept_attn_map.t()) 
    return G.div(a * b)


def interpolate_cross_attention_maps(attention_probs_list, gram_reg=False, min_seq=True):
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
        if gram_reg:
            interp_ca_map = gram_matrix(interp_ca_map)
        else:
            interp_ca_map = interp_ca_map.permute(0, 2, 3, 1)            
        interp_attention_maps.append(interp_ca_map)
    interpolated_attention_maps = torch.stack(interp_attention_maps)
    return interpolated_attention_maps

def get_concept_attn_map(attn_maps, bsz, word_ids):
    concept_attn_maps = []
    for i in range(bsz):
        word_id = word_ids[i]
        for ca_map in attn_maps:
            concept_attn_maps.append(ca_map[i, :, :, word_id])
    return concept_attn_maps

def calculate_pairwise_mse_loss(attn_maps_1, attn_maps_2, with_gram=False, reduction='mean'):
    mse_losses = []
    for map1, map2 in zip(attn_maps_1, attn_maps_2):
        # check that map1 and map2 have same shape (batch_size, seq_len, seq_len, feature_dim)
        if map1.shape != map2.shape:
            raise ValueError("Mismatch in attention map shapes: {} vs {}".format(map1.shape, map2.shape))
        if with_gram:
            map1 = gram_matrix(map1.permute(0, 3, 1, 2))
            map2 = gram_matrix(map2.permute(0, 3, 1, 2))
            mse_loss = F.mse_loss(map1, map2, reduction=reduction) # torch.sqrt()
        else:
            mse_loss = F.mse_loss(map1, map2, reduction=reduction) # / (map1.shape[-1] - 1)
        mse_losses.append(mse_loss)
    mean_mse_loss = torch.mean(torch.stack(mse_losses))
    return mean_mse_loss

# Condition functions for unet hook: which layers to save
def get_names_of_ca_q_k_blocks(name, filter_names, with_ca_maps=True):
    # return names of cross-attn query and key layers, all blocks 
    return any(f in name for f in filter_names)

def get_names_of_sa_v_mid_block(name, filter_names, with_ca_maps=False):
    # return names of self-attn values layers, only middle block 
    # with_ca_maps: flag to save cross-attn layers
    return (
        (any(f in name for f in filter_names) if with_ca_maps else False) or
        'mid_block' in name and 
        'attn1' in name and 
        (
            'to_v' in name or 
            'to_v_lora' in name
        )
    )

def get_names_of_sa_v_all_blocks(name, filter_names, with_ca_maps=False):
    # return names of self-attn values layers, all blocks 
    # with_ca_maps: flag to save cross-attn layers
    return (
        (any(f in name for f in filter_names) if with_ca_maps else False) or
        'attn1' in name and 
        (
            'to_v' in name or 
            'to_v_lora' in name
        )
    )

def get_names_return_none(name, filter_names, with_ca_maps=False):
    return None
