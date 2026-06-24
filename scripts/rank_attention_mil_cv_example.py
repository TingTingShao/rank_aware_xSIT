"""Run 5-fold bag-level cross-validation for rank-attention MIL.

This script keeps the original ``rank_attention_mil_example.py`` untouched and
reuses its dataset conversion, model, and ranking helpers.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rank_attention_mil_example import (  # noqa: E402
    BAG_ID_COL,
    BAG_LABEL_COL,
    INST_ID_COL,
    generate_rankings,
    make_mil_arrays_from_dataframe,
    make_model,
)


BASE_PATH = (
    "/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/"
    "multiinstanceEmbryoRanking/analysis"
)

DATASETS = {
    "remove": "data/cleaned_full_dataset_remove.csv",
    "negative": "data/cleaned_full_dataset_negative.csv",
}

N_SPLITS = 5
RANDOM_STATE = 42


def make_pid_outcome(df):
    pid_outcome = df[[BAG_ID_COL, BAG_LABEL_COL]].drop_duplicates()
    duplicated_pids = pid_outcome[BAG_ID_COL].duplicated(keep=False)
    if duplicated_pids.any():
        bad_pids = pid_outcome.loc[duplicated_pids, BAG_ID_COL].unique()[:10]
        raise ValueError(
            f"Some PIDs have more than one {BAG_LABEL_COL!r} value: {bad_pids}"
        )
    return pid_outcome.reset_index(drop=True)


def split_dataframe_by_pid(df, train_pids, test_pids):
    train_mask = df[BAG_ID_COL].isin(train_pids)
    test_mask = df[BAG_ID_COL].isin(test_pids)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def add_bag_predictions(rankings, bag_probs, group_sizes):
    bag_preds = np.argmax(bag_probs, axis=1)
    rankings["bag_prediction"] = np.repeat(bag_preds, group_sizes)
    rankings["bag_probability_0"] = np.repeat(bag_probs[:, 0], group_sizes)
    rankings["bag_probability_1"] = np.repeat(bag_probs[:, 1], group_sizes)
    rankings["bag_livebirth_probability"] = rankings["bag_probability_1"]
    return rankings


def add_test_metadata(rankings, df_test_sorted):
    metadata = df_test_sorted[[BAG_ID_COL, INST_ID_COL, BAG_LABEL_COL]].copy()
    metadata["instance"] = (
        metadata[BAG_ID_COL].astype(str)
        + "_embryo"
        + metadata[INST_ID_COL].astype(str)
    )
    return rankings.merge(
        metadata,
        on="instance",
        how="left",
        validate="one_to_one",
    )


def run_fold(dataset_name, fold_id, df, train_pids, test_pids):
    df_train, df_test = split_dataframe_by_pid(df, train_pids, test_pids)

    X, y, group_sizes, instance_y, instance_names, _, bag_X = make_mil_arrays_from_dataframe(
        df_train
    )
    (
        X_te,
        y_te,
        group_sizes_te,
        instance_y_te,
        instance_names_te,
        df_test_sorted,
        bag_X_te,
    ) = make_mil_arrays_from_dataframe(df_test)

    print(
        f"\n[{dataset_name}] fold {fold_id}",
        f"train_bags={len(group_sizes)}",
        f"test_bags={len(group_sizes_te)}",
        f"train_pos={int(y.sum())}/{len(y)}",
        f"test_pos={int(y_te.sum())}/{len(y_te)}",
    )

    model = make_model()
    model.fit(X, y, group_sizes, instance_y=instance_y, bag_X=bag_X)

    attention_logits = model.predict_attention_logits(
        X_te, group_sizes_te, grouped=True, bag_X=bag_X_te
    )
    attention_weights = model.predict_attention_weights(
        X_te, group_sizes_te, grouped=True, bag_X=bag_X_te
    )
    bag_probs = model.predict_proba(X_te, group_sizes_te, bag_X=bag_X_te)

    rankings = generate_rankings(
        group_sizes_te,
        instance_names_te,
        instance_y_te,
        attention_logits,
        attention_weights,
    )
    rankings = add_test_metadata(rankings, df_test_sorted)
    rankings = add_bag_predictions(rankings, bag_probs, group_sizes_te)
    rankings["dataset"] = dataset_name
    rankings["fold"] = fold_id

    return rankings

def run_dataset(dataset_name, dataset_path):
    df = pd.read_csv(dataset_path)
    pid_outcome = make_pid_outcome(df)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rankings = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(pid_outcome[BAG_ID_COL], pid_outcome[BAG_LABEL_COL]),
        start=1,
    ):
        train_pids = pid_outcome.iloc[train_idx][BAG_ID_COL]
        test_pids = pid_outcome.iloc[test_idx][BAG_ID_COL]
        fold_rankings.append(
            run_fold(dataset_name, fold_id, df, train_pids, test_pids)
        )

    all_rankings = pd.concat(fold_rankings, ignore_index=True)
    output_path = os.path.join(
        BASE_PATH,
        f"test_attention_rankings_{dataset_name}_5fold.csv",
    )
    all_rankings.to_csv(output_path, index=False)
    print(f"\nSaved {dataset_name} 5-fold rankings to {output_path}")
    return all_rankings


def main():
    for dataset_name, relative_path in DATASETS.items():
        dataset_path = os.path.join(BASE_PATH, relative_path)
        run_dataset(dataset_name, dataset_path)


if __name__ == "__main__":
    main()


