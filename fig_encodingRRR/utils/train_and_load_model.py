
import os
import pandas as pd

from utils.utils import make_folder, remove_space, log_kv
from utils.RRRGD_main_CV import train_model_hyper_selection
from utils.RRRGD import RRRGD_model

def _get_model_folder(data_folder, *args):
    subfolder = ""
    for arg in args:
        subfolder += f"{arg}"
    data_folder = make_folder(os.path.join('./model', data_folder, remove_space(subfolder)))
    return data_folder


def _get_RRR_fname(gp, RRRGD_params, spliti=None):
    model_folder = _get_model_folder(gp['data_id'], gp['X_inc'], gp['y_inc'], gp['vl'])
    if spliti is None:
        RRRGD_fname = os.path.join(model_folder, remove_space(f"RRRGD_{RRRGD_params}")) 
    else:
        RRRGD_fname = os.path.join(model_folder, remove_space(f"RRRGD_{RRRGD_params}_split{spliti}")) 
    log_kv(RRRGD_fname=RRRGD_fname)
    return RRRGD_fname

def _train_RRReids(Xy_regression_eids, RRRGD_fname, RRRGD_params):
    if os.path.isfile(RRRGD_fname):
        pass
    else:
        train_model_hyper_selection(Xy_regression_eids, model_fname=RRRGD_fname,
                        **RRRGD_params)

"""
train the RRRGD model by considering neurons from *all areas*
- model saved in the data folder: f"Data_bwm/{data_id}/{Xy_inc_criteria}+{var_list}"
- RRRGD model: 
    - name: RRRGD2_{RRRGD_params}.pt
"""
def get_RRRglobal_res(Xy_regression, gp):
    RRRGD_params = gp['RRRGDglobal_p']
    RRRGD_fname = _get_RRR_fname(gp, RRRGD_params)
    log_kv(RRRGD_fname=RRRGD_fname)
    Xy_regression_eids = Xy_regression
    _train_RRReids(Xy_regression_eids, RRRGD_fname, RRRGD_params)


def _loadres_RRReids(res_eids, res_dict_temp, mname):
    for eid in sorted(list(res_eids.keys())):
        res = res_eids[eid]
        N = len(res["r2"])
        res_dict_temp['eid'] += [eid] * N
        res_dict_temp['ni'] += list(range(N))
        res_dict_temp[f"{mname}_r2"] += res["r2"].tolist()
        res_dict_temp[f"{mname}_beta"] += [res["model"]["beta"][ni] for ni in range(N)] # (ncoef, T)
        res_dict_temp[f"{mname}_U"] += [res["model"]["U"][ni] for ni in range(N)] # (ncoef, ncomp)
        res_dict_temp[f"{mname}_V"] += [res["model"]["V"]] * N # (ncomp, T)
        res_dict_temp[f"{mname}_b"] += [res["model"]["b"][ni][0] for ni in range(N)] # (T)
        res_dict_temp[f"{mname}_ncomp"] += [res["model"]["best_ncomp"]] * N
        res_dict_temp[f"{mname}_ncomp_bias"] += [res["model"]["best_ncomp_bias"]] * N
        res_dict_temp[f"{mname}_l2"] += [res["model"]["best_l2"]] * N
    return res_dict_temp

"""
load the RRRGD model of *multiple areas*
"""
def load_RRRglobal_res(gp, spliti=None):
    mname = "RRRglobal"

    columns = ["eid", "ni", f"{mname}_beta", f"{mname}_r2"] 
    columns += [f"{mname}_U", f"{mname}_V", f"{mname}_b"]
    columns += [f"{mname}_ncomp", f"{mname}_ncomp_bias", f"{mname}_l2"]
    res_dict_temp = {c: [] for c in columns}

    RRRGD_fname = _get_RRR_fname(gp, gp['RRRGDglobal_p'], spliti=spliti)
    if os.path.isfile(f"{RRRGD_fname}.pt"):
        # Use the same RRRGD_model.params2RRRre for both RRRGD_model and RRRGD_model_Vdep
        if 'Vdep' in gp['RRRGDglobal_p']:
            res_all = RRRGD_model.params2RRRres(RRRGD_fname, Vdep=gp['RRRGDglobal_p']['Vdep'], has_area=False) 
        else:
            res_all = RRRGD_model.params2RRRres(RRRGD_fname, has_area=False) 
    else:
        assert False, f"no file {RRRGD_fname}"

    res_dict_temp = _loadres_RRReids(res_all, res_dict_temp, mname)
    RRR_res_df = pd.DataFrame.from_dict(res_dict_temp)

    return RRR_res_df
