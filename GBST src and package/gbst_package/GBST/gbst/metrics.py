
import numpy as np
from sklearn.metrics import roc_auc_score

def evalauc(preds, dtrain):
    # print("Preds shape:", preds.shape)
    labels = dtrain.get_label().astype(int)
    y_arr = np.zeros([preds.shape[0], preds.shape[1]])
    for i, label in enumerate(labels):
        y_arr[i, :label] = 1
        y_arr[i, label:] = 0
    hazards = 1./(1.+np.exp(-preds))
    mults = np.ones(hazards.shape[0])
    auc_total = []
    for timestep in range(0, hazards.shape[1]):
        mults = mults * (1 - hazards[:, timestep])
        try:
            auc = roc_auc_score(y_true=y_arr[:, timestep], y_score=mults)
            auc_total.append(auc)
        except Exception as e:
            # If all candidates
            pass
    return 'AUC', float(np.sum(auc_total)) / len(auc_total)
