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


def make_dict(path):
    ti_evaluate_cache = DistributedCache(path)
    d = {k: {**v, **aggregate_similarities(v)} for k, v in ti_evaluate_cache.get().items()}
    medium_base_res = {}
    for key in d.keys():
        if live_object_data[d[key]['config']['class_name']] == 'live':
            kt = 'live_text_similarity'
            ki = 'live_image_similarity'
        else:
            kt = 'object_text_similarity'
            ki = 'object_image_similarity'

        kb = 'base_image_similarity'
        if not d[key][kt] or not d[key][ki]:
            if live_object_data[d[key]['config']['class_name']] == 'live':
                kt = 'live_with_class_text_similarity'
                ki = 'live_with_class_image_similarity'
            else:
                kt = 'object_with_class_text_similarity'
                ki = 'object_with_class_image_similarity'
            kb = 'base_with_class_image_similarity'
        medium_base_res[key] = {
            'TS': np.round(d[key][kt], 6),
            'BIS': np.round(d[key][kb], 6),
            'IS': np.round(d[key][ki], 6),
        }
    return medium_base_res

device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
device

method_names = ['TI', 'DB'] 

res_values = []

ts_vals = []
is_vals = []
bis_vals = []

for method_name in method_names:
    path ="/home/mdnikolaev/aschumachenko/diffusers/examples/"
    if method_name == 'TI':
        base_path = path + "textual_inversion/res_TI/"
    else:
        base_path = path + "dreambooth/res_DB/"
    exps_list = ['0-res-dog6_', '0-res-backpack_dog_', '0-res-berry_bowl_'] 
    exps = [exp + method_name for exp in exps_list]
    concepts = [exp.split('0-res-')[1].split('_' + method_name)[0] for exp in exps]
    res = make_dict(base_path + '*/eval.cache')

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

colors = ['r', 'b']
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
