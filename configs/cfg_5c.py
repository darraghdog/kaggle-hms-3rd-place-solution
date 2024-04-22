import os
import sys
from importlib import import_module
import platform
import json
import numpy as np
import copy
import glob
from types import SimpleNamespace

if not platform.node().isdigit():
    if platform.system()=='Darwin':
        PATH = '/Users/darraghhanley/Documents/kaggle-hms'
    else:
        PATH = './'
    os.chdir(f'{PATH}')
    
import configs.cfg_5a as cfg

cfg = copy.deepcopy(cfg.cfg)
cfg.spec_args_center = dict(sample_rate=200, n_fft=2048, n_mels=128//2, f_min=0.53, f_max=30, win_length=16, hop_length=4)
cfg.aug_drop_spec_prob = 0.7