import os
import sys
from importlib import import_module
import platform
import json
import numpy as np
import glob
from types import SimpleNamespace

if not platform.node().isdigit():
    if platform.system()=='Darwin':
        PATH = '/Users/darraghhanley/Documents/kaggle-hms'
    else:
        PATH = './'
    os.chdir(f'{PATH}')

    sys.path.append("configs")
    sys.path.append("augs")
    sys.path.append("models")
    sys.path.append("data")
    sys.path.append("postprocess")

    from default_config import basic_cfg
    import pandas as pd
    cfg = basic_cfg
    cfg.debug = True
    if platform.system()!='Darwin':
        cfg.name = os.path.basename(__file__).split(".")[0]
        cfg.output_dir = f"{PATH}/weights/{os.path.basename(__file__).split('.')[0]}"
    cfg.data_dir = f"{PATH}/datamount/"
    cfg.data_folder = f'{cfg.data_dir}/train_eegs/'
    cfg.data_folder_spec = f'{cfg.data_dir}/train_spectrograms/'
else:

    from default_config import basic_cfg
    import pandas as pd
    cfg = basic_cfg
    cfg.debug = True
    PATH = "/mount/hms"
    cfg.name = os.path.basename(__file__).split(".")[0]
    cfg.output_dir = f"{PATH}/models/{os.path.basename(__file__).split('.')[0]}"
    cfg.data_dir = f"{PATH}/data/"
    cfg.data_folder = f'{cfg.data_dir}/hms-harmful-brain-activity-classification/train_eegs/'


cfg.train_df = f'{cfg.data_dir}/train_folded_6k6c.csv'

cfg.val_df = f'{cfg.data_dir}/train_folded_6k6.csv'
cfg.targets = ['seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']

# stages
cfg.test = False
cfg.test_data_folder = cfg.data_folder
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1

#logging
cfg.neptune_project = "common/quickstarts"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"

#model
cfg.model = "mdl_1"
cfg.mixup_spectrogram = False
cfg.mixup_signal = False
cfg.mixup_beta = 1.
#cfg.backbone = "tf_efficientnetv2_s.in21k_ft_in1k"
cfg.backbone = "mixnet_xl"
cfg.pool = "gem"
cfg.gem_p_trainable = True
cfg.pretrained = True
cfg.in_channels = 1
cfg.spec_args = dict(sample_rate=200, n_fft=1024, n_mels=128, f_min=0.53, f_max=40, win_length=128, hop_length=39)
cfg.model_args = dict(drop_rate=0.2, drop_path_rate=0.2)


# OPTIMIZATION & SCHEDULE
cfg.n_landmarks = 16
cfg.fold = 0
cfg.epochs = 12
cfg.lr = 0.0012
cfg.optimizer = "Adam"
cfg.weight_decay = 0.
cfg.clip_grad = 20.
cfg.warmup = 0
cfg.batch_size = 8
cfg.mixed_precision = False # True
cfg.pin_memory = False
cfg.grad_accumulation = 4.
cfg.num_workers = 8

# DATASET
cfg.dataset = "ds_1a"
cfg.normalization = 'image'

#EVAL
cfg.calc_metric = False
cfg.simple_eval = False
# augs & tta

# Postprocess
cfg.post_process_pipeline =  "pp_dummy"
cfg.metric = "default_metric"
# augs & tta

#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True

cfg.aug_drop_spec_prob = 0.5
cfg.aug_drop_spec_max = 8
cfg.aug_bandpass_prob = 0.2
cfg.aug_bandpass_max = 8
cfg.enlarge_len = 0
