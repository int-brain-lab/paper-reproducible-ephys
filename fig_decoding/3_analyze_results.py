import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from reproducible_ephys_functions import LAB_MAP


def process_results(varis, re_regions, res_path, save_path, score_threshold=-0.05):
    """
    Process variations, filter data, and save results to Parquet files.

    Parameters:
        varis (list): List of variable names.
        res_path (str or Path): Path to results directory.
        re_regions (list): List of regions to process.
        save_path (str or Path): Path to save processed Parquet files.
        score_threshold (float): Threshold below which EIDs are filtered out.

    Returns:
        None
    """
    filter_out_eids = []
    df_per_var = []

    for var in varis:
        df_list = []
        for fold_idx in range(1, 6):  # Loop over folds 1 to 5
            # Load configurations
            config_path = Path(res_path) / var / "multi-region-reduced_rank" / f"fold_{fold_idx}" / "configs.npy"
            configs = np.load(config_path, allow_pickle=True)
            res_dict = {}

            # Process each region
            for region in re_regions:
                res_dict[region] = {}
                path = Path(res_path) / var / "multi-region-reduced_rank" / f"fold_{fold_idx}" / region
                for fname in os.listdir(path):
                    if os.path.isdir(path / fname):
                        continue
                    eid = fname[:-4]
                    data = np.load(path / fname, allow_pickle=True).item()
                    tmp = data["test_metric"]

                    # Adjust by chance level for specific variables
                    if var in ["choice", "stimside", "reward"]:
                        tmp_chance_lvl = data["test_chance_metric"]
                        tmp = tmp - tmp_chance_lvl

                    res_dict[region][eid] = tmp

            # Create a DataFrame and reshape it
            df = pd.DataFrame(res_dict).reset_index()
            df_melted = pd.melt(df, id_vars=['index'], var_name='region', value_name='score')
            df_melted.columns = ["pid", "region", "score"]

            # Add unit count information
            df_melted["unitcount"] = None
            for config in configs:
                mask = (df_melted["pid"] == config["eid"]) & (df_melted["region"] == config["region"])
                if not mask.any():
                    continue
                row_idx = df_melted[mask].index.item()
                df_melted.loc[row_idx, "unitcount"] = config["n_units"]

            df_list.append(df_melted)

        # Merge data and calculate mean scores
        merged_df = pd.concat(df_list)
        df = merged_df.groupby(["pid", "region"]).agg({"score": 'mean', "unitcount": "first"}).reset_index()

        # Filter out EIDs with outlier scores
        filter_out_eids += list(df.loc[df["score"] < score_threshold, "pid"])
        df_per_var.append(df)

    # Remove duplicate EIDs
    filter_out_eids = list(set(filter_out_eids))

    # Save each variable's processed DataFrame to a Parquet file
    for i, var in enumerate(varis):
        df = df_per_var[i]
        for _eid in filter_out_eids:
            df.drop(df[df["pid"] == _eid].index, inplace=True)
        output_path = Path(save_path) / f"{var}.parquet"
        df.to_parquet(output_path, engine="pyarrow")


def evaluate_f1_with_permutation(unit, varis, index, b, one, n_permutations=5000, random_state=42):
    """
    Evaluate F1 score with permutation testing using scipy.stats.permutation_test.

    Parameters:
        varis (list): A list of variable names.
        index (int): Index to select the variable from `varis`.
        b (dict): A mapping to decode lab information from paths.
        one (object): An object with method `eid2path` to retrieve paths based on `eid`.
        n_permutations (int): Number of permutations for the test.
        random_state (int): Seed for controlling random number generation (default is 42).

    Returns:
        dict: Observed F1 score and p-value from the permutation test.
    """
    # Load and preprocess the data
    vari = varis[index]
    print(f"Processing variable: {vari}")
    data_file = f'{vari}.parquet'
    d = pd.read_parquet(data_file)

    # Add 'lab' and 'subject' columns based on paths
    eids = [one.pid2eid(pid)[0] for pid in d['pid'].values]
    pths = one.eid2path(eids)
    d['lab'] = [b[str(p).split('/')[6]] for p in pths]
    d['subject'] = [str(p).split('/')[8] for p in pths]

    # Drop rows with missing values in specified columns
    d = d.dropna(subset=['score', 'lab', 'region', 'subject'])

    # Prepare feature matrix X and target vector y
    X = np.c_[d.score.to_numpy(), d.unitcount.to_numpy()]
    y = d.region.to_numpy() if unit == "region" else d.lab.to_numpy()

    print(f"Compare {unit} for variable {vari}:")

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def compute_f1(y_true, y_pred):
        """Helper function to compute the mean F1 score across folds."""
        f1_cv = []
        for train_idx, test_idx in skf.split(X, y_true):
            clf = make_pipeline(StandardScaler(), KNeighborsClassifier())
            clf.fit(X[train_idx], y_true[train_idx])
            pred = clf.predict(X[test_idx])
            f1 = f1_score(y_true[test_idx], pred, average='macro')
            f1_cv.append(f1)
        return np.mean(f1_cv)

    # Compute the observed F1 score
    observed_f1 = compute_f1(y, y)
    print(f"Observed F1 score: {observed_f1}")

    # Permutation test
    def statistic(y_true, y_permuted):
        return compute_f1(y_permuted, y_true)

    result = permutation_test(
        data=(y, y),
        statistic=statistic,
        vectorized=False,
        n_resamples=n_permutations,
        random_state=random_state,
        alternative='greater'
    )

    p_value = result.pvalue
    print(f"Permutation test p-value: {p_value}")

    return observed_f1, p_value


def run_permutation_tests(varis, units, b, one, n_permutations=10):
    """
    Run permutation tests for F1 score for the given units and variables.

    Parameters:
        varis (list): List of variable names.
        units (list): List of units to evaluate (e.g., "region", "lab").
        b (dict): Mapping to decode lab information from paths.
        one (object): ONE instance for data retrieval.
        n_permutations (int): Number of permutations for the test.

    Returns:
        dict: Dictionaries for F1 scores and p-values for each unit.
    """
    results = {unit: {"f1_dict": {}, "pval_dict": {}} for unit in units}

    for unit in units:
        for vari_idx, vari in enumerate(varis):
            print(f"Running permutation test for unit: {unit}, variable: {vari}")
            f1_score, p_value = evaluate_f1_with_permutation(
                unit, varis, vari_idx, b, one, n_permutations=n_permutations
            )
            results[unit]["f1_dict"][vari] = f1_score
            results[unit]["pval_dict"][vari] = p_value

    return results


def save_results_to_files(results, output_dir="./"):
    """
    Save F1 scores and p-values dictionaries to pickle files.

    Parameters:
        results (dict): Results dictionary containing F1 scores and p-values.
        output_dir (str): Directory to save the pickle files.
    """
    for unit, data in results.items():
        with open(f"{output_dir}/{unit}_f1_dict.pkl", "wb") as f:
            pickle.dump(data["f1_dict"], f)
        with open(f"{output_dir}/{unit}_pval_dict.pkl", "wb") as f:
            pickle.dump(data["pval_dict"], f)
    print("Results saved successfully.")



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", type=str)
    args = ap.parse_args()

    ONE.setup(
        base_url="https://openalyx.internationalbrainlab.org", silent=True,
        cache_dir=args.base_path
    )
    one = ONE(password='international')

    _, b, lab_cols = LAB_MAP()
    re_regions = ["PO", "LP", "DG", "CA1", "VISa"]
    varis = ['choice', 'stimside', 'reward', 'wheel-speed']

    dec_d = {
        'stimside': 'stimside', 
        'choice': 'choice',
        'feedback': 'feedback', 
        'wheel-speed': 'wheel-speed'
    }   

    print("Processing decoding results: ")

    process_results(varis, re_regions, res_path=args.base_path, save_path="./")

    print("Running permutation tests... This may take approximately 20 minutes.")
    units = ["region", "lab"]
    results = run_permutation_tests(varis, units, b, one, n_permutations=5000)

    print("Saving results:")
    save_results_to_files(results, output_dir="./")

