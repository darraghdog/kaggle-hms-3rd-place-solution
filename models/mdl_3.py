from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import timm
from torch.distributions import Beta


'''
self = Net(cfg)
'''

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")

class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Z=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            if len(Y.shape) == 1:
                Y = coeffs * Y + (1 - coeffs) * Y[perm]
            else:
                Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
                
        if Z:
            return X, Y, Z

        return X, Y

mirror_ids = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
              [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
              [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15],
              [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15],
              [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15],
              [0,1,2,3,4,6,7,8,9,10,11,12,13,14,15],
              [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15],
              [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15],
              [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15],
              [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15],
              [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15],
              [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15],
              [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15],
              [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15],
              [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15],
              [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]

s0 = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fp1', 'Fp2','F3', 'F4', 'C3', 'C4', 'P3', 'P4']
s1 = ['F7', 'F8',   'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2']   
m = [[2, 3, 4, 5, 6, 7, 14, 15, 10, 11, 12, 13, 14, 15, 6, 7],
[8, 9, 0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13]]
np.array(m).transpose().tolist()
mirror_ids2 = [[2, 8],
                 [3, 9],
                 [4, 0],
                 [5, 1],
                 [6, 2],
                 [7, 3],
                 [14, 4],
                 [15, 5],
                 [10, 0],
                 [11, 1],
                 [12, 8],
                 [13, 9],
                 [14, 10],
                 [15, 11],
                 [6, 12],
                 [7, 13]]

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Net(nn.Module):

    def __init__(self, cfg: Any):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = len(self.cfg.targets)
        self.preprocessing = nn.Sequential(MelSpectrogram(**cfg.spec_args),AmplitudeToDB())     
        self.backbone = timm.create_model(cfg.backbone, 
                                          pretrained=cfg.pretrained, 
                                          num_classes=0, 
                                          global_pool="", 
                                          in_chans=self.cfg.in_channels,
                                          **cfg.model_args
                                         )
        self.backbone.conv_stem = torch.nn.Conv2d(self.cfg.in_channels, 32, 
                                                  kernel_size=cfg.conv_stem_k_size, 
                                                  stride=cfg.conv_stem_stride, 
                                                  padding=cfg.conv_stem_padding, 
                                                  bias=False)

        self.mixup = Mixup(cfg.mixup_beta)
        self.mixup_signal = cfg.mixup_signal
        self.mixup_spectrogram = cfg.mixup_spectrogram
        if 'efficientnet' in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.num_features

        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(backbone_out, self.n_classes)
        self.loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        print('model params: ',count_parameters(self))
        
    def forward(self, batch):

        x = batch['input'].permute(0,2,1).float()
        y = batch["target"].float()
        
        if (self.training) & ( self.mixup_signal ):
            x,y = self.mixup(x,y)
            
        bs, c, l = x.shape
        x = x.reshape(bs*c,l)
        
        with torch.cuda.amp.autocast(enabled=False): 
            x = self.preprocessing(x)
        bsc, h, w = x.shape
        x = x.reshape(bs,c,h,w)
                
        x1 = x - x[:,mirror_ids].mean(2)
        x2 = x - x[:,mirror_ids2].mean(2)
        x = torch.stack([x,x1, x2],dim=2)
        
        x = torch.cat((x[:,::2], x[:,1::2]), 1)
        
        if (self.training):
            for tt in range(x.shape[0]):
                if self.cfg.aug_drop_spec_prob > np.random.random():
                    drop_ct = np.random.randint(1, 1+self.cfg.aug_drop_spec_max)
                    drop_idx = np.random.choice(np.arange(x.shape[1]), drop_ct)
                    x[tt, drop_idx] = 0.

        x = x.transpose(2,1)
        x = x.reshape(bs,self.cfg.in_channels,c*h,w)

        if (self.training) & (self.mixup_spectrogram):
            x,y = self.mixup(x,y)
        
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x[:,:,0,0]

        logits = self.head(x)
        loss = self.loss_fn(F.log_softmax(logits, dim=1),y)
        outputs = {}
        outputs['loss'] = loss
        if not self.training:
            outputs["logits"] = logits
 
        return outputs
