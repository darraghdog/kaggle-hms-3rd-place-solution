import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.signal import butter, lfilter
import random

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def butter_bandpass_filter(data, high_freq=20, low_freq=0.5, sampling_rate=200, order=2):
    nyquist = 0.5 * sampling_rate
    high_cutoff = high_freq / nyquist
    low_cutoff = low_freq / nyquist
    b, a = butter(order, [low_cutoff,high_cutoff], btype='band', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

tr_collate_fn = torch.utils.data.dataloader.default_collate
val_collate_fn = torch.utils.data.dataloader.default_collate

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.df = df.copy()
        print(f'mode:{mode}, df.shape:{df.shape}')
        
        if mode == "train":
            filt_low = min([i[0] for i in cfg.vote_ct_ranges])
            filt_hi = max([i[1] for i in cfg.vote_ct_ranges])
            filtidx = (self.df.filter(like='_vote').sum(1)>=filt_low ) & \
                        (self.df.filter(like='_vote').sum(1)<=filt_hi)
            self.df = self.df[filtidx]
            print(f'filtered mode:{mode}, self.df.shape:{self.df.shape}')
            self.df['vote_ct'] = self.df.filter(like='_vote').sum(1).values
            self.df['label_weight']  = 0.
            
            step_decay = 1 - ((cfg.curr_epoch / cfg.epochs) * cfg.vote_ct_weight_decay)
            vote_ct_weights = [ max(w * step_decay, mw)  if t+1!= len(cfg.vote_ct_weights) else w \
                           for  t,(w,mw) in enumerate(zip(cfg.vote_ct_weights, cfg.vote_ct_weights_min))]
            print(f'Vote count weights : {vote_ct_weights}')

            # step_decay = 1 - (((1 + cfg.curr_epoch) / cfg.epochs) * cfg.vote_ct_weight_decay)
            # vote_ct_weights = [ w * step_decay  if t+1!= len(cfg.vote_ct_weights) else w for  t,w in enumerate(cfg.vote_ct_weights)]
            
            for (filt_low, filt_hi), wt in zip(cfg.vote_ct_ranges, vote_ct_weights):
                filtidx = (self.df.filter(like='_vote').sum(1)>=filt_low ) & \
                            (self.df.filter(like='_vote').sum(1)<=filt_hi)
                self.df.loc[filtidx, 'label_weight'] = wt
        
        self.mode = mode
        self.aug = aug
        self.data_folder = cfg.data_folder
        
        targets = self.df[cfg.targets].values
        
        self.s0 = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fp1', 'Fp2','F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        self.s1 = ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2']     
        self.df[cfg.targets] = self.df[cfg.targets].values / self.df[cfg.targets].values.sum(axis=1,keepdims=True)
        
        self.eegs = self.df['eeg_id'].values
        
    def __getitem__(self, idx):

        #for idx in range(50):
        row = self.df.iloc[idx]
            
        eeg_id, eeg_label_offset_seconds = row[['eeg_id','eeg_label_offset_seconds']].astype(int)

        y = row[self.cfg.targets].values.astype(np.float32)
        
        eeg, center = self.load_one(eeg_id, eeg_label_offset_seconds)
        #print(f"{eeg.mean():0.4f}  {eeg.std():0.4f}")

        feature_dict = {
            "input": torch.from_numpy(eeg),
            #"label_weight": torch.tensor(row.label_weight),
            "center":torch.tensor(center, dtype = torch.long),
            "target":torch.from_numpy(y)
        }
        if self.mode == "train":
            feature_dict["label_weight"] = torch.tensor(row.label_weight)
            
        return feature_dict

    def __len__(self):
        return len(self.eegs)
    
    def load_one(self, eeg_id, eeg_label_offset_seconds=0):
        
        eeg_combined = pd.read_parquet(f'{self.data_folder}{eeg_id}.parquet')
        
        start = int(200*eeg_label_offset_seconds)
        win_len = 10000

        if self.mode == "train":
            start_shifted = int(np.random.uniform(start - win_len//3, start + win_len//3))
            start_shifted = np.clip(start_shifted, 0, eeg_combined.shape[0] - win_len)
            
        else:
            start_shifted = start
        shift = start - start_shifted
        
        eeg = eeg_combined.iloc[start_shifted:start_shifted+win_len]
        eeg_1 = eeg[self.s0].values 
        eeg_2 = eeg[self.s1].values
        
        x = (eeg_1 - eeg_2)
        x[np.isnan(x)] = 0
        
        if self.mode == "train":
            if np.random.random()>0.5:
                x = x[::-1].copy()
            if np.random.random()>0.5:
                x[:,np.arange(x.shape[-1])[1::2]], x[:,np.arange(x.shape[-1])[0::2]] = \
                    x[:,np.arange(x.shape[-1])[0::2]], x[:,np.arange(x.shape[-1])[1::2]]
        
        x = butter_bandpass_filter(x, high_freq=self.cfg.butter_high_freq, low_freq=self.cfg.butter_low_freq, order=self.cfg.butter_order)
        if (self.mode == "train") and (self.cfg.aug_bandpass_prob > np.random.random() ):
            filt_idx = np.random.choice(np.arange(x.shape[-1]), 1 + np.random.randint(self.cfg.aug_bandpass_max))
            high_freq_aug = np.random.randint(10, self.cfg.butter_high_freq)
            low_freq_aug = np.random.uniform(self.cfg.butter_low_freq, 2)
            x[:,filt_idx] = butter_bandpass_filter(x[:,filt_idx], high_freq=high_freq_aug, low_freq=low_freq_aug, order=max(self.cfg.butter_order, 1))
                               
        x = x.clip(-1024, 1024)
        x /= 32
        
        center = shift + win_len//2
        return x, center
