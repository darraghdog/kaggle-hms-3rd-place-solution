import numpy as np
from sklearn.metrics import f1_score
import torch
import scipy as sp

def get_score(y_true, y_pred):
    return 0
#     score = sp.stats.pearsonr(y_true, y_pred)[0]
#     return score





def calc_metric(cfg, pp_out, val_df, pre="val"):
    return 0
# #     if isinstance(pred_df,list):
# #         pred_df,gt_df = pred_df
# #     else:
# #         gt_df = None

#     y_true = val_df['score'].values
#     y_pred = val_data['preds'].cpu().numpy()
#     score = get_score(y_true.flatten(), y_pred.flatten())
# #     print(score)

# #     df['score'] = df['location'].apply(ast.literal_eval)
# #     df['span'] = df['location'].apply(location_to_span)
# #     spans_true = df['span'].values

# #     df_pred = pred_df.copy()
# #     # df_pred['location'] = df_pred['location'].apply(ast.literal_eval)
# #     df_pred['span'] = df_pred['pred_location'].apply(pred_location_to_span)
# #     spans_pred = df_pred['span'].values

# #     score = span_micro_f1(spans_pred, spans_true)

#     if hasattr(cfg, "neptune_run"):
#         cfg.neptune_run[f"{pre}/score/"].log(score, step=cfg.curr_step)
#         print(f"{pre} score: {score:.6}")
# #     else:
# #         return score

# #     if gt_df is not None:
# #         df_pred = gt_df.copy()
# #         df_pred['span'] = df_pred['pred_location'].apply(pred_location_to_span)
# #         spans_pred = df_pred['span'].values

# #         score = span_micro_f1(spans_pred, spans_true)

# #         if hasattr(cfg, "neptune_run"):
# #             cfg.neptune_run[f"{pre}/score_debug/"].log(score, step=cfg.curr_step)
# # #             print(f"{pre} score_debug: {score:.6}")          
#     return score

