# import sys
from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    ipython.magic("config Completer.use_jedi = True")
    ipython.magic("matplotlib inline")

import warnings
warnings.filterwarnings("ignore")

# import regex
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
# import random
import numpy as np
# from matplotlib.ticker import FormatStrFormatter
import torch
import torch.backends.cuda
import matplotlib_inline
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg')

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from nb_utils.clip_eval import aggregate_similarities
from nb_utils.cache import DistributedCache, Cache
from nb_utils.experiments_viewer import ExpsViewer
from nb_utils.clip_eval import ExpEvaluator
# from nb_utils.images_viewer import MultifolderViewer
from nb_utils.configs import live_object_data

def make_dict(path):
    ti_evaluate_cache = DistributedCache(path)
    d = {k: {**v, **aggregate_similarities(v)} for k, v in ti_evaluate_cache.get().items()}
    medium_base_res = {}
    for key in d.keys():
        if live_object_data[d[key]['config']['class_name']] == 'live':
            kt = 'live_text_similarity'
            ki = 'live_image_similarity'
            kt_std = 'live_text_similarity_std'
            ki_std = 'live_image_similarity_std'
        else:
            kt = 'object_text_similarity'
            ki = 'object_image_similarity'
            kt_std = 'object_text_similarity_std'
            ki_std = 'object_image_similarity_std'
        kb = 'base_image_similarity'
        kd = 'dino_image_similarity'

        if not d[key][kt] or not d[key][ki]:
            if live_object_data[d[key]['config']['class_name']] == 'live':
                kt = 'live_with_class_text_similarity'
                ki = 'live_with_class_image_similarity'
                kt_std = 'live_with_class_text_similarity_std'
                ki_std = 'live_with_class_image_similarity_std'

            else:
                kt = 'object_with_class_text_similarity'
                ki = 'object_with_class_image_similarity'
                kt_std = 'object_with_class_text_similarity_std'
                ki_std = 'object_with_class_image_similarity_std'
            kb = 'base_with_class_image_similarity'
            kd = 'dino_with_class_image_similarity'

        medium_base_res[key] = {
            'TS': np.round(d[key][kt], 6),
            'BIS': np.round(d[key][kb], 6),
            'DIS': np.round(d[key][kd], 6),
            'IS': np.round(d[key][ki], 6),
            'TS_std': np.round(d[key][kt_std], 6),
            'IS_std': np.round(d[key][ki_std], 6),
        }
    return medium_base_res


device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

evaluator = ExpEvaluator(device)

dir_path = "/home/jovyan/shares/SR006.nfs2/aschumachenko/aschumachenko"
base_path = f"{dir_path}/core/res_CR/"
eval_path = f'{base_path}*/eval.cache'
concepts = ['bear_plushie', 'colorful_sneaker'] # 'dog6', 'cat', 'teapot', 'grey_sloth_plushie', 'shiny_sneaker'] # ['dog6', 'cat2', 'duck_toy', 'teapot', 'cat']    #  'dog6', 'cat2', 'duck_toy', 'teapot', 'dog6', 'clock', 'cat', 'backpack', 'cat2', 'berry_bowl', 'red_cartoon', 'poop_emoji', 'dog', 'monster_toy', 'grey_sloth_plushie', 'vase', 'dog5'
method_name = 'CR_sdxl'  # 'CR'/'CD'
chp = 300
exps_nums = range(0, 7) # [0] # range(17, 23) # [25, 26] # range(10, 18)

exps = [f'00{i}-res-{concept}_{method_name}' for i in exps_nums for concept in concepts]
# [a for a in os.listdir(base_path)]

exps_viewer = ExpsViewer(
    base_path=base_path,
    exp_filter_fn=lambda x: x in list(exps),
    ncolumns=7,
    lazy_load=True,
    evaluator=evaluator
)

stats = exps_viewer.evaluate(
    exps_names=exps,
    checkpoint_idx=f'{chp}',
    inference_specs=('25', '5.0')
)

for key, value in stats.items():
    exp_cache = Cache(os.path.join(value['config']['output_dir'], 'eval.cache'))
    exp_cache.update({key: value})
    print('ready')

ti_evaluate_cache = DistributedCache(path_template=eval_path)
# ti_evaluate_cache.keys()
# ti_evaluate_cache[key]
# print(ti_evaluate_cache)
cache_data = ti_evaluate_cache.get()

# Ensure aggregate_similarities can handle None values appropriately
d = {k: {**v, **aggregate_similarities(v)} for k, v in cache_data.items()}
# print(d[('0015-res-cat_CR', ('1000', '50', '7.0'))].keys())
# d[key]

res = make_dict(eval_path)
print(res)

# запустив код, можно посмотреть сохраненные картинки, если они были правильно сохранены
# ti_exps_viewer = ExpsViewer(
#     # Base path to experiments
#     f'{dir_path}/diffusers/examples/core/res_CR',
#     # Filter experiments
#     # exp_filter_fn=lambda x: x in [x for x in os.listdir('results') if "CD" not in x],
#     # Filter prompts
#     # filter_fn='a photo of a sks',
#     # Defines number of columns in experiment selector
#     ncolumns=4,
#     # Whether to load all images only on demand
#     lazy_load=True,
#     # ExpEvaluator instance to perform evaluation. See scripts/*/inference.sh
#     evaluator=None,
# )
# ti_exps_viewer.view()
