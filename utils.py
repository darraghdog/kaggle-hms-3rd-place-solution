
import random
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DataLoader, WeightedRandomSampler
from torch import nn, optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import lr_scheduler
import importlib
import math
import neptune
from neptune.utils import stringify_unsupported

import logging
import pickle



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles= 0.5, last_epoch= -1):
    """
    from https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)




def calc_grad_norm(parameters,norm_type=2.):
    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None
        
    return total_norm

def sync_across_gpus(t, world_size):
    torch.distributed.barrier()
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_model(cfg, ds):
    Net = importlib.import_module(cfg.model).Net
    net = Net(cfg)
    return net

def create_checkpoint(cfg, model, optimizer, epoch, scheduler=None, scaler=None):

    
    state_dict = model.state_dict()
    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

def get_dataset(df, cfg, mode='train'):
    
    #modes train, val, index
    print(f"Loading {mode} dataset")
    
    if mode == 'train':
        dataset = get_train_dataset(df, cfg)
    elif mode == 'val':
        dataset = get_val_dataset(df, cfg)
    else:
        pass
    return dataset

def get_dataloader(ds, cfg, mode='train'):
    
    if mode == 'train':
        dl = get_train_dataloader(ds, cfg)
    elif mode =='val':
        dl = get_val_dataloader(ds, cfg)
    return dl


def get_train_dataset(train_df, cfg):

    train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    if cfg.data_sample > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(cfg.data_sample))
    return train_dataset


def get_train_dataloader(train_ds, cfg):

    try:
        if cfg.random_sampler_frac > 0:
            
            num_samples = int(len(train_ds) * cfg.random_sampler_frac)
            sample_weights = train_ds.sample_weights
            sampler = WeightedRandomSampler(sample_weights, num_samples= num_samples )
        else:
            sampler = None
    except:
        sampler = None

    train_dataloader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataset(val_df, cfg, allowed_targets=None):
    val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")
    return val_dataset


def get_val_dataloader(val_ds, cfg):

    sampler = SequentialSampler(val_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader



def get_optimizer(model, cfg):

    params = model.parameters()

    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return optimizer



def get_scheduler(cfg, optimizer, total_steps):

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) // cfg.world_size,
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
    )
    return scheduler


def setup_neptune(cfg):
    
    
    neptune_run = neptune.init_run(
        project=cfg.neptune_project,
        tags=cfg.tags,
        mode=cfg.neptune_connection_mode,
        capture_stdout=False,
        capture_stderr=False,
        source_files=[f'models/{cfg.model}.py',f'data/{cfg.dataset}.py',f'configs/{cfg.name}.py'],
        flush_period=cfg.flush_period
    )


    neptune_run["cfg"] = stringify_unsupported(cfg.__dict__)

    return neptune_run


def get_data(cfg):

    # setup dataset

    print(f"reading {cfg.train_df}")
    df = pd.read_csv(cfg.train_df)

    if cfg.test:
        test_df = pd.read_csv(cfg.test_df)
    else:
        test_df = None
        
    if cfg.val_df is not None:
        df2 = pd.read_csv(cfg.val_df)
        if cfg.fold == -1:
            val_df = df2[df2["fold"] == 0]
        else:
            val_df = df2[df2["fold"] == cfg.fold]    
    else:        
        if cfg.fold == -1:
            val_df = df[df["fold"] == 0]
        else:
            val_df = df[df["fold"] == cfg.fold]
        
    train_df = df[df["fold"] != cfg.fold]
        
    return train_df, val_df, test_df


def flatten(t):
    return [item for sublist in t for item in sublist]

def set_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows',10000)
    pd.set_option('display.width', 10000)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)


