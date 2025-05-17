import sys
from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    ipython.magic("config Completer.use_jedi = True")
    ipython.magic("matplotlib inline")

import warnings
warnings.filterwarnings("ignore")

import regex
import os
import random
import numpy as np
from matplotlib.ticker import FormatStrFormatter


import torch
import torch.backends.cuda

import matplotlib_inline
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg')

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from nb_utils.clip_eval import aggregate_similarities

from nb_utils.cache import DistributedCache, Cache

from nb_utils.experiments_viewer import ExpsViewer
from nb_utils.clip_eval import ExpEvaluator
from nb_utils.images_viewer import MultifolderViewer
from nb_utils.configs import live_object_data


# def make_dict(path):
#     ti_evaluate_cache = DistributedCache(path)
#     d = {k: {**v, **aggregate_similarities(v)} for k, v in ti_evaluate_cache.get().items()}
#     medium_base_res = {}
#     for key in d.keys():
#         if live_object_data[d[key]['config']['class_name']] == 'live':
#             kt = 'live_text_similarity'
#             ki = 'live_image_similarity'
#         else:
#             kt = 'object_text_similarity'
#             ki = 'object_image_similarity'

#         kb = 'base_image_similarity'
#         if not d[key][kt] or not d[key][ki]:
#             if live_object_data[d[key]['config']['class_name']] == 'live':
#                 kt = 'live_with_class_text_similarity'
#                 ki = 'live_with_class_image_similarity'
#             else:
#                 kt = 'object_with_class_text_similarity'
#                 ki = 'object_with_class_image_similarity'
#             kb = 'base_with_class_image_similarity'
#         medium_base_res[key] = {
#             'TS': np.round(d[key][kt], 6),
#             'BIS': np.round(d[key][kb], 6),
#             'IS': np.round(d[key][ki], 6),
#         }
#     return medium_base_res


device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

method_names = ['TI', 'DB', 'CR'] 

res_values = []

ts_vals = []
is_vals = []
bis_vals = []


# res_TI = [
#     {('0001-res-dog6_TI', ('500', '50', '7.0')): {'TS': 0.269, 'BIS': 0.869, 'IS': 0.804}},
#     {('0001-res-backpack_dog_TI', ('500', '50', '7.0')): {'TS': 0.250, 'BIS': 0.737, 'IS': 0.660}},
#     {('0001-res-berry_bowl_TI', ('500', '50', '7.0')): {'TS': 0.247, 'BIS': 0.692, 'IS': 0.652}}
# ]

# res_DB = [
#     {('0001-res-dog6_DB', ('500', '50', '7.0')): {'TS': 0.253, 'BIS': 0.919, 'IS': 0.873}},
#     {('0001-res-backpack_dog_DB', ('500', '50', '7.0')): {'TS': 0.242, 'BIS': 0.763, 'IS': 0.679}},
#     {('0001-res-berry_bowl_DB', ('500', '50', '7.0')): {'TS': 0.242, 'BIS': 0.784, 'IS': 0.705}}
# ]

# res_CR = [
#     {('0001-res-dog6_CR', ('400', '50', '7.0')): {'TS': 0.255166, 'BIS': 0.953125, 'IS': 0.885781}},
#     {('0001-res-backpack_dog_CR', ('600', '50', '7.0')): {'TS': 0.235435, 'BIS': 0.805176, 'IS': 0.692266}},
#     {('0001-res-berry_bowl_CR', ('200', '50', '7.0')): {'TS': 0.225601, 'BIS': 0.821777, 'IS': 0.769102}}
# ]

# res_ea = [
#     {('00001-res-dog6_CR', ('4000', '50', '7.0')): {'TS': 0.238725, 'BIS': 0.98291, 'IS': 0.859863}}, 
#     {('00005-res-cat_CR', ('4000', '50', '7.0')): {'TS': 0.258025, 'BIS': 0.941406, 'IS': 0.729261}}
# ]

# res_e = [
#     {('00002-res-dog6_CR', ('4000', '50', '7.0')): {'TS': 0.230096, 'BIS': 0.982422, 'IS': 0.869629}},
#     {('00006-res-cat_CR', ('4000', '50', '7.0')): {'TS': 0.249582, 'BIS': 0.944824, 'IS': 0.764674}} 
# ]

# res_a = [
#     {('00003-res-dog6_CR', ('4000', '50', '7.0')): {'TS': 0.231946, 'BIS': 0.981934, 'IS': 0.865851}},
#     {('00007-res-cat_CR', ('4000', '50', '7.0')): {'TS': 0.253232, 'BIS': 0.949219, 'IS': 0.738461}}
# ]

# res_n = [
#     {('00004-res-dog6_CR', ('4000', '50', '7.0')): {'TS': 0.226505, 'BIS': 0.979492, 'IS': 0.876902}},
#     ('00008-res-cat_CR', ('4000', '50', '7.0')): {'TS': 0.23487, 'BIS': 0.963867, 'IS': 0.818693},   
# ]


for method_name in method_names:
    path ="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/examples/"
    exps_list = ['0001-res-dog6_', '0001-res-backpack_dog_', '0001-res-berry_bowl_'] 
    exps = [exp + method_name for exp in exps_list]
    concepts = [exp.split('0001-res-')[1].split('_' + method_name)[0] for exp in exps]
    if method_name == 'TI':
        base_path = path + "textual_inversion/res_TI/"
        res = res_TI
    elif method_name == 'DB':
        base_path = path + "dreambooth/res_DB/"
        res = res_DB
    else:
        base_path = path + "core/res_CR/"
        res = res_CR
    # res = make_dict(base_path + '*/eval.cache')
    

    for keys, values in res.items():
        res_values.append(values)

    for entry in res_values:
        ts_vals.append(entry['TS'])
        is_vals.append(entry['IS'])
        bis_vals.append(entry['BIS'])


def reshape_func(data):
    new_data = data[3:]
    n = len(new_data) // 2
    data_list = [new_data[i:i + n] for i in range(0, len(new_data), n)]
    return np.array(data_list)

ts_vals = reshape_func(ts_vals)
print(ts_vals)
is_vals = reshape_func(is_vals)
bis_vals = reshape_func(bis_vals) 

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(12, 8)) 

colors = ['r', 'b', 'g']
for j in range(2):
    for i, exp in enumerate(concepts):
        for k, method_name in enumerate(method_names):
            color = colors[k] 
            if j == 0:
                axes[j, i].plot(ts_vals[k, i], is_vals[k, i], marker='o', linestyle=' ', color=color, markersize=10, alpha=0.5)
                axes[j, i].set_ylabel('image similarity', fontsize=16)
            else:
                axes[j, i].plot(ts_vals[k, i], bis_vals[k, i], marker='o', linestyle=' ', color=color, markersize=10, alpha=0.5)
                axes[j, i].set_ylabel('base image similarity', fontsize=16)
            axes[j, i].set_title(f'{exp}', fontsize=18)
            axes[j, i].set_xlim(0.2, 0.3)
            axes[j, i].set_ylim(0.6, 1.0)
            axes[j, i].tick_params(axis='both', which='major', labelsize=14)
            axes[j, i].set_xlabel('text similarity', fontsize=16)
            axes[j, i].grid(True)

for ax in axes.flatten():
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # ax.legend(method_names, loc='upper right', fontsize=14)


fig.legend(method_names, loc='lower center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=len(method_names), fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(f'{path}/text_image_similarity.png') 
plt.show()
