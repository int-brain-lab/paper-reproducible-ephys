
import numpy as np
import os, pdb
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class CustomStratifiedGroupKFold:
    def __init__(self, n_splits=5, prioritize_large_groups=True):
        self.n_splits = n_splits
        self.prioritize_large_groups = prioritize_large_groups

    def split(self, X, y, groups):
        group_sizes = {group: np.sum(groups == group) for group in np.unique(groups)}
        sorted_groups = sorted(group_sizes, key=group_sizes.get, reverse=self.prioritize_large_groups)
        
        # Reorder the dataset based on sorted groups
        group_indices = {group: np.where(groups == group)[0] for group in sorted_groups}
        ordered_indices = np.concatenate([group_indices[group] for group in sorted_groups])

        X_ordered, y_ordered, groups_ordered = X[ordered_indices], y[ordered_indices], groups[ordered_indices]

        skf = StratifiedGroupKFold(n_splits=self.n_splits)
        return skf.split(X_ordered, y_ordered, groups_ordered)

def make_CV_prediction(coef_vs, _ys, trainpred_func,
                       split="stratifiedGroup", eid_all=None, n_splits=3,):
    sgkf = CustomStratifiedGroupKFold(n_splits=n_splits)
    splits = sgkf.split(coef_vs, _ys, groups=eid_all)
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
                N_y=len(np.unique(_ys)),
                test_idxs=test_idxs, test_pred=test_pred,) 


def trainpred_func_SVC(X_train, y_train, X_test):
    clf = make_pipeline(StandardScaler(), SVC())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, None, clf
