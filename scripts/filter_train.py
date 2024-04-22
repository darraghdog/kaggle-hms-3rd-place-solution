import pandas as pd
import numpy as np
from tqdm import tqdm

def filter_train(train, filter_ct,  drop_dupes = False):
    
    train = train[train[label_columns].sum(1)>filter_ct].copy()

    #group all eeg_id with same votes
    train['eeg_id_l'] = train[['eeg_id'] + label_columns].astype(str).agg('_'.join, axis=1)

    #for each unique eeg label combination calculate all overlaping time frames of same label and take the one row that has the highest overlap to all 
    rows = []
    for eeg_id_l in tqdm(train['eeg_id_l'].unique()):
        df0 = train[train['eeg_id_l']==eeg_id_l].reset_index(drop=True).copy()
        offsets = df0['spectrogram_label_offset_seconds'].astype(int).values
        x = np.zeros(offsets.max()+600)
        for o in offsets:
            x[o:o+600] += 1
        best_idx = np.argmax([x[o:o+600].sum() for o in offsets])
        rows += [df0.iloc[best_idx]]

    df = pd.DataFrame(rows)
    #some (although few) eeg have multiple labels, just take the first one
    if not drop_dupes:
        return df
    
    df2 = df.drop_duplicates(subset='eeg_id').copy()
    return df2



label_columns = ['seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote']

train = pd.read_csv('datamount/train_folded.csv')

# Golden set for validation
dfout = filter_train(train.copy(), filter_ct = 8,  drop_dupes = True)
dfout.to_csv('datamount/train_folded_6k6.csv', index=False)

# Regular training set
dfout = filter_train(train.copy(), filter_ct = 8,  drop_dupes = False)
dfout.to_csv('datamount/train_folded_6k6c.csv', index=False)

# Large training set with low count votes
dfout = filter_train(train.copy(), filter_ct = 0,  drop_dupes = False)
dfout.to_csv('datamount/train_folded_6k6c_1up.csv', index=False)





