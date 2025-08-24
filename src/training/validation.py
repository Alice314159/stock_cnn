
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "val_acc": accuracy_score(y_true, y_pred),
        "val_f1_macro": f1_score(y_true, y_pred, average='macro'),
        "val_mcc": matthews_corrcoef(y_true, y_pred)
    }
