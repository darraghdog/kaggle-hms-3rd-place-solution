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
    
import configs.cfg_2a as cfg_2a

cfg = copy.deepcopy(cfg_2a.cfg)
cfg.butter_order = 2