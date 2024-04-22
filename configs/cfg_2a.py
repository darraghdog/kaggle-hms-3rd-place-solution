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


cfg.train_df = f'{cfg.data_dir}/train_folded_6k6c_1up.csv'
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
cfg.model = "mdl_2"
cfg.mixup_spectrogram = False
cfg.mixup_signal = False
cfg.mixup_beta = 1.
# cfg.backbone = 'efficientvit_b2.r288_in1k'
cfg.pool = "gem"
cfg.gem_p_trainable = True
cfg.pretrained = True
cfg.in_channels = 1
cfg.spec_args = dict(sample_rate=200, n_fft=1024, n_mels=128, f_min=0.53, f_max=40, win_length=128, hop_length=39)
cfg.model_args = dict(drop_rate=0.2)#, drop_path_rate=0.2)

cfg.feat_dim = 256
cfg.feat_init_ksize = 21

cfg.n_landmarks = 16

cfg.subsampling_rate_pre = 1
cfg.subsampling_rates = [2,2,2,2]
cfg.conv_k_sizes = [7,7,5,5]

sampdiv = np.product((cfg.subsampling_rates))
cfg.max_len = int(np.ceil(10000/sampdiv))
encoder_config = SimpleNamespace(**{})
encoder_config.input_dim=128
encoder_config.encoder_dim=128
encoder_config.num_layers=3
encoder_config.num_attention_heads= 4
encoder_config.feed_forward_expansion_factor=1
encoder_config.conv_expansion_factor= 2
encoder_config.input_dropout_p= 0.1#0.1
encoder_config.feed_forward_dropout_p= 0.2#0.1
encoder_config.attention_dropout_p= 0.2#0.1
encoder_config.conv_dropout_p= 0.2#0.1
encoder_config.conv_kernel_size= 51
encoder_config.reduce_layer_indices = []
cfg.encoder_config = encoder_config

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 30
cfg.eval_epochs = 2
cfg.lr = 0.001
cfg.optimizer = "Adam"
cfg.weight_decay = 0.
cfg.clip_grad = 4.
cfg.warmup = 0
cfg.batch_size = 64
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 3.
cfg.num_workers = 8


# DATASET
cfg.dataset = "ds_2"
cfg.normalization = 'image'
cfg.vote_ct_ranges = [[0, 2], [3, 8], [8, 9999]]
cfg.vote_ct_weights = [0.3, 0.5, 1.0]
cfg.vote_ct_weight_decay = 1.2
cfg.vote_ct_weights_min = [0.02, 0.02, 1.0]
cfg.reload_train_loader = True
cfg.curr_epoch = 0

'''
for i in range(cfg.epochs):
    cfg.curr_epoch = i
    step_decay = 1 - ((cfg.curr_epoch / cfg.epochs) * cfg.vote_ct_weight_decay)
    weights = [ max(w * step_decay, mw)  if t+1!= len(cfg.vote_ct_weights) else w \
                   for  t,(w,mw) in enumerate(zip(cfg.vote_ct_weights, cfg.vote_ct_weights_min))]
    print(i, weights)
'''

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
cfg.shift_freq = False
cfg.butter_order = 1

cfg.butter_high_freq = 30
cfg.butter_low_freq = 1.6

