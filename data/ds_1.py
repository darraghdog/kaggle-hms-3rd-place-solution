import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.signal import butter, lfilter

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

eeg_cols =         ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
eeg_cols_flipped = ['Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'Fz', 'Cz', 'Pz', 'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1']
flip_map = dict(zip(eeg_cols,eeg_cols_flipped))

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.df = df.copy()
        print(f'mode:{mode}, df.shape:{df.shape}')
        
        self.mode = mode
        self.aug = aug
        self.data_folder = cfg.data_folder
        
        targets = self.df[cfg.targets].values
        
        self.s0 = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fp1', 'Fp2','F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        self.s1 = ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2']     
        self.df[cfg.targets] = self.df[cfg.targets].values / self.df[cfg.targets].values.sum(axis=1,keepdims=True)
        
        self.eegs = self.df['eeg_id'].values
        
    def __getitem__(self, idx):

        row = self.df.iloc[idx]
            
        eeg_id, eeg_label_offset_seconds = row[['eeg_id','eeg_label_offset_seconds']].astype(int)

        y = row[self.cfg.targets].values.astype(np.float32)
        
        eeg, center = self.load_one(eeg_id, eeg_label_offset_seconds)

        feature_dict = {
            "input": torch.from_numpy(eeg),
            "center":torch.tensor(center, dtype = torch.long),
            "target":torch.from_numpy(y)
        }
        return feature_dict

    def __len__(self):
        return len(self.eegs)
    
    def load_one(self, eeg_id, eeg_label_offset_seconds=0):
        eeg_combined = pd.read_parquet(f'{self.data_folder}{eeg_id}.parquet')
        start = int(200*eeg_label_offset_seconds)
        win_len = 10000

        if self.mode == "train":
            if np.random.rand() < 0.5:
                eeg_combined = eeg_combined.rename(columns=flip_map).copy()
            start_shifted = int(np.random.uniform(start - win_len//3, start + win_len//3))
            start_shifted = np.clip(start_shifted, 0, eeg_combined.shape[0] - win_len)
        else:
            start_shifted = start
        shift = start - start_shifted
        
        eeg = eeg_combined.iloc[start_shifted:start_shifted+win_len]
        x = (eeg[self.s0].values - eeg[self.s1].values)
        
        x[np.isnan(x)] = 0
        
        x = butter_bandpass_filter(x)
        if self.mode == "train":
            if self.cfg.aug_bandpass_prob > np.random.random():
                filt_idx = np.random.choice(np.arange(x.shape[-1]), 1 + np.random.randint(self.cfg.aug_bandpass_max))
                high_freq_aug = np.random.randint(10, 20)
                low_freq_aug = np.random.uniform(0.0001, 2)
                x[:,filt_idx] = butter_bandpass_filter(x[:,filt_idx], high_freq=high_freq_aug, low_freq=low_freq_aug)
        
        x = x.clip(-1024, 1024)
        x /= 32
        
        center = shift + win_len//2
        
        if self.mode == "train":
            if np.random.rand() < 0.7:
                center = int( np.random.uniform(center - self.cfg.enlarge_len // 3, 
                                                center + self.cfg.enlarge_len // 3))
        
        return x, center
