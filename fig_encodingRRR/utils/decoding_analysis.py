
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_CV_prediction(coef_vs, _ys, trainpred_func,
                       split="stratifiedGroup", eid_all=None, n_splits=3,):
    if split == "stratifiedGroup":
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        # groups
        splits = sgkf.split(coef_vs, _ys, groups=eid_all)
    elif split == "stratified":
        sgkf = StratifiedKFold(n_splits=n_splits)
        splits = sgkf.split(coef_vs, _ys)
    f1s = []; accs = []; 
    test_idxs = []; test_pred = []; 
    for train, test in splits:
        y_pred, y_lowd, clf = trainpred_func(coef_vs[train], _ys[train], coef_vs[test])
        accs.append(balanced_accuracy_score(_ys[test], y_pred))
        f1s.append(f1_score(_ys[test], y_pred, average='macro'))
        test_idxs.append(test)
        test_pred.append(y_pred)
    f1 = np.mean(f1s)
    acc = np.mean(accs)
    test_idxs = np.concatenate(test_idxs)
    test_pred = np.concatenate(test_pred)

        
    return dict(perf=dict(f1=f1, acc=acc, f1s=f1s, accs=accs), 
                test_idxs=test_idxs, test_pred=test_pred,) 


def trainpred_func_SVC(X_train, y_train, X_test):
    clf = make_pipeline(StandardScaler(), SVC(class_weight='balanced'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, None, clf
