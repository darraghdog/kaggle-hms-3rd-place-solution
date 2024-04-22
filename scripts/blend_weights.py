import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import importlib, os, glob, copy, sys
from utils import set_pandas_display
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
os.chdir('/Users/darraghhanley/Documents/kaggle-hms')
sys.path.append("configs")
sys.path.append("metrics")
sys.path.append("models")
sys.path.append("data")

set_pandas_display()
def load_preds(cfg_name, FOLD):
    cfg = importlib.import_module(cfg_name)
    cfg = copy.copy(cfg.cfg)
    cfg.fold = FOLD
    ds = importlib.import_module(cfg.dataset)
    df = pd.read_csv(cfg.val_df)
    test_ds = ds.CustomDataset(df.query(f'fold=={FOLD}'), cfg, cfg.val_aug, mode="valid")
    val_df = test_ds.df
    
    assert len(glob.glob(f'weights/{cfg_name}/fold{FOLD}/val*')) == 2, glob.glob(f'weights/{cfg_name}/fold{FOLD}/val*')
    val_data_name = glob.glob(f'weights/{cfg_name}/fold{FOLD}/val*')[0]
    val_data = torch.load(val_data_name, map_location=torch.device('cpu'))
    pp = importlib.import_module(cfg.post_process_pipeline)
    pp_out = pp.post_process_pipeline(cfg, val_data, val_df)
    pred_df = val_df[['eeg_id', 'fold']]
    pred_df.loc[:,[i.replace('_vote', '_logits') for i in cfg.targets]] = pp_out['logits'].numpy()
    return val_df, pred_df


weights_ls = "cfg_1 cfg_2a cfg_2b cfg_3 cfg_4 cfg_5a cfg_5b cfg_5c cfg_5d" .split()
 

# Load the validation file form one of the configs
cfg_name = weights_ls[0]
cfg = importlib.import_module(cfg_name)
cfg = copy.copy(cfg.cfg)
dfs = [load_preds(cfg_name, F) for F in range(4)]
val_df = pd.concat([d[0] for d in dfs])


X = []
for cfg_name in weights_1d + weights_mel_ls:
    dfs = [load_preds(cfg_name, F) for F in range(4)]
    pred_df1 = pd.concat([d[1] for d in dfs])
    logits = pred_df1.filter(like='_logits').values
    logits = torch.from_numpy(logits)
    X.append(logits)


X = torch.stack(X, -1)
targets = val_df[cfg.targets].values
targets = targets / targets.sum(axis=1,keepdims=True)
y = torch.from_numpy(targets)



# set the device to cpu
device = torch.device("cpu")

num_params = X.shape[-1]
# define the model
class Net(nn.Module):
    def __init__(self, n_models = 23):
        super(Net, self).__init__()
        self.fc = nn.Linear(n_models, 1, bias = False) 
        self.fc_c = torch.nn.Parameter(torch.zeros(6)[None,:])
        self.fc.load_state_dict({'weight': torch.ones(n_models)[None,:]/n_models})
    def forward(self, x):
        return self.fc(x).squeeze(-1) + self.fc_c

# create regularization
def lasso_reg(model, lambda_lasso):
    lasso_loss = 0
    for param in model.parameters():
        lasso_loss += torch.sum(torch.abs(param))
    return lambda_lasso * lasso_loss

# number of training iterations
num_epochs = 801
# KFold cross-validation
n_splits = 5  # define the number of folds you want.
LR = 0.1
L_REGULARISATION = 0.0
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
weight_ls = []
params_ls = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f'Fold {fold+1}')

    # create a model and move it to CPU
    model = Net(X.shape[-1])
    model.to(device)
    
    # loss function
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    # optimization (stochastic gradient descent)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    
    # prepare train and validation data loaders
    train_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X.float()[train_idx], y[train_idx]), batch_size=2048//2, shuffle=True, drop_last = True)
    val_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X.float()[val_idx], y[val_idx]), batch_size=2048//2, drop_last = True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10 * len(train_data), gamma=0.95)
    
    for epoch in range(num_epochs):
        for inputs, labels in train_data:
            # move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            logits = model(inputs)
            loss = loss_fn(F.log_softmax(logits, dim=1), labels)
            # lasso regularization
            loss += lasso_reg(model, L_REGULARISATION)

            # backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # adjust learning rate
            scheduler.step()
  
        if (epoch) % 100 == 0:
            # forward pass
            logits = model(X.float()[val_idx]).squeeze(-1)
            val_loss = loss_fn(F.log_softmax(logits, dim=1), y[val_idx])
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{num_epochs}, Val loss: {val_loss.item():0.5f}, LR: {lr:0.6f}')
    weight_ls.append(model.fc.weight.detach().numpy().copy())
    params_ls.append(model.fc_c.detach().numpy().copy())

weights = np.stack(weight_ls)[:,0]
params_ls = np.concatenate(params_ls)[:,:]

print(f'Correlation of model weights')
print(np.corrcoef(weights).round(4))

print(f'Correlation of class offsets')
print(np.corrcoef(params_ls).round(4))
 
 
# Save the weights
weights_names = weights_1d + weights_mel_ls
weights_dict = weights.mean(0).astype(np.float32)

weights_dict = {n:i for n,i in zip(weights_names, weights_dict)}

sum(weights_dict.values())

# Blend weights score
X_wtd = (X * torch.from_numpy(weights.mean(0))[None,None,:]).sum(-1)
loss_fn(F.log_softmax(X_wtd, dim=1), y)

# Add offset and check the score
X_wtd2 = X_wtd + params_ls.mean(0)[None,:]
loss_fn(F.log_softmax(X_wtd2, dim=1), y)

torch.corrcoef(X.transpose(2,1).reshape(-1, 6).permute(1,0))


print('Final config weights')
print(weights_dict)

print('Final class bias')
print(weights_dict)



