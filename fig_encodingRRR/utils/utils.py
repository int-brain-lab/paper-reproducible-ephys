import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
from scipy import signal
import matplotlib.pyplot as plt
import os, pickle, pdb
import pandas as pd

def remove_space(s):
    s = s.replace(" ","")
    s = s.replace("'", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace(":", "")
    s = s.replace(",", "_")
    return s

def log_kv(**kwargs):
    print(f"{kwargs}")

def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder

def get_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device


def np2tensor(v):
    return torch.from_numpy(v)

def np2param(v, grad=True):
    return nn.Parameter(np2tensor(v), requires_grad=grad)

def tensor2np(v):
    return v.cpu().detach().numpy()


def compute_R2_main(y, y_pred, clip=True):
    """
    :y: (K, T, N) or (K*T, N)
    :y_pred: (K, T, N) or (K*T, N)
    """
    N = y.shape[-1]
    if len(y.shape) > 2:
        y = y.reshape((-1, N))
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, N))
    r2s = np.asarray([r2_score(y[:, n], y_pred[:, n]) for n in range(N)])
    if clip:
        return np.clip(r2s, 0., 1.)
    else:
        return r2s


# typically is dict of {area: {eid: xx}}
def load_or_save_dict(_fname, _main, **params):
    if os.path.isfile(_fname):
        with open(_fname, 'rb') as f:
            res = pickle.load(f)
    else:
        res = _main(**params)
        with open(_fname, 'wb') as f:
            pickle.dump(res, f)
    return res

# typically is a df with columns area, eid, ni
# * important: cells should not contain nparray
def load_or_save_df(_fname, _main, **params):
    if os.path.isfile(_fname):
        df = pd.read_csv(_fname)
    else:
        df = _main(**params)
        df.to_csv(_fname, index=False,)
    return df


def load_neuron_nparray(df, column, rows_mask=None):
    if rows_mask is None:
        vs = df[column]
    else:
        vs = df.loc[rows_mask, column]
    vs = np.array([v for v in vs])
    return vs


def find_bestdelay_byCC(beh, act, plot=False):
    """
    :beh: (K,T)
    :act: (K,T,N)
    """
    beh = beh - np.mean(beh, axis=1, keepdims=True)
    act = act - np.mean(act, axis=1, keepdims=True)
    lags = signal.correlation_lags(beh.shape[1], act.shape[1], mode="valid")
    CCs = []
    for ni in range(act.shape[-1]):
        CC = np.mean([signal.correlate(beh[k], act[k,:,ni], mode="valid") for k in range(beh.shape[0])], 0)
        CCs.append(CC)
    CCs = np.asarray(CCs)
    CC_agg = np.linalg.norm(CCs, axis=0)
    best_delay = lags[np.argmax(CC_agg)]
    if (best_delay>np.min(lags)) and (best_delay<np.max(lags)):
        success=True
    else:
        best_delay = 0
        success=False
    if plot:
        plt.figure(figsize=(3,3))
        plt.plot(lags, CC_agg)
        plt.axvline(x=best_delay)
        plt.show()
    return best_delay, success, CCs


