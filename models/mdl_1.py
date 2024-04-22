from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import timm
from torch.distributions import Beta
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

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
        '''
            torch.Size([128, 128, 257])
        '''
        x = self.preprocessing(x)
        bsc, h, w = x.shape
        x = x.reshape(bs,c,h,w)
        
        if (self.training):
            for tt in range(x.shape[0]):
                if self.cfg.aug_drop_spec_prob > np.random.random():
                    drop_ct = np.random.randint(1, 1+self.cfg.aug_drop_spec_max)
                    drop_idx = np.random.choice(np.arange(x.shape[1]), drop_ct)
                    x[tt, drop_idx] = 0.
        x = x.reshape(bs,1,c*h,w)

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
