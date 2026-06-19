"""Rank MIL instances by attention scores guided by instance labels.

This example builds a tiny flattened MIL dataset with bag labels and partial
instance labels.

Training uses one joint objective:

    bag-level classification BCE
    + lambda_rank * pairwise RankNet loss on raw attention logits
    + lambda_inst * optional pointwise instance-logit BCE

The gradient tree encoder and the attention neural network are learned together
through ``GradBoostingClassifier.fit``. After fitting, we derive an instance
ranking inside each bag from both raw logits and post-softmax attention weights.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sit.gradtree.grad_boosting import GradBoostingClassifier


CYCLE_FEATURES = [
    "female_age",
    "cycle_num",
    "n_2pn",
    "freeze_all",
    "usable_embryos",
    "pituitary_inhibition_agonist-long",
    "pituitary_inhibition_agonist-short",
    "pituitary_inhibition_antagonist",
    "pituitary_inhibition_none",
]

DAY3_FEATURES = [
    "d3_n_blastomeres",
    "d3_fragmentation_0",
    "d3_fragmentation_1",
    "d3_fragmentation_2",
    "d3_fragmentation_3",
    "d3_fragmentation_4",
    "d3_symmetry_0",
    "d3_symmetry_1",
    "d3_symmetry_2",
]

DAY56_FEATURES = [
    "expansion_EB",
    "expansion_EiB",
    "expansion_ExB",
    "expansion_HB",
    "expansion_HiB",
    "expansion_M",
    "expansion_VEB",
    "expansion_abnormal",
    "icm_-1",
    "icm_A",
    "icm_B",
    "icm_C",
    "blastocyst_-1",
    "blastocyst_I",
    "blastocyst_II",
    "blastocyst_III",
    "trophectoderm_-1",
    "trophectoderm_1",
    "trophectoderm_2",
    "trophectoderm_3",
    "assess_day",
]

BAG_LABEL_COL = "livebirth"
BAG_ID_COL = "PID"
INST_ID_COL = "embryo_nr"
INST_LABEL_COL = "embryo_label"
TRANSFER_ORDER_COL = "transfer_order"


def make_mil_arrays_from_dataframe(
    df,
    feature_fill_value=0.0
):
    """Convert one-embryo-per-row dataframe into GradBoostingClassifier inputs.

    Returns
    -------
    X : np.ndarray, shape (n_embryos, n_features)
        Flattened embryo feature matrix.
    y : np.ndarray, shape (n_bags,)
        One livebirth label per PID.
    group_sizes : np.ndarray, shape (n_bags,)
        Number of embryo rows per PID, in the same order as y.
    instance_y : np.ndarray, shape (n_embryos,)
        Embryo labels aligned with X. NaN values are allowed and ignored by the
        current instance losses.
    instance_names : np.ndarray, shape (n_embryos,)
        Readable embryo ids for printing rankings.
    df_sorted : pd.DataFrame
        Sorted dataframe aligned with X and instance_y.
    """
    # features = CYCLE_FEATURES + DAY3_FEATURES + DAY56_FEATURES
    features = CYCLE_FEATURES + DAY56_FEATURES
    # features = DAY3_FEATURES + DAY56_FEATURES

    required_cols = [
        BAG_ID_COL,
        BAG_LABEL_COL,
        INST_ID_COL,
        INST_LABEL_COL,
        *features,
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    input_n_rows = len(df)
    input_n_bags = df[BAG_ID_COL].nunique(dropna=False)
    sort_cols = [BAG_ID_COL, INST_ID_COL]
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True).copy()

    if len(df_sorted) != input_n_rows:
        raise ValueError(
            f"Row count changed during sorting: {input_n_rows} -> {len(df_sorted)}"
        )

    grouped = df_sorted.groupby(BAG_ID_COL, sort=False, dropna=False)
    output_n_bags = grouped.ngroups
    if output_n_bags != input_n_bags:
        raise ValueError(
            f"Bag count changed during grouping: {input_n_bags} -> {output_n_bags}. "
            f"Check missing values in {BAG_ID_COL!r}."
        )

    bag_label_counts = grouped[BAG_LABEL_COL].nunique(dropna=False)
    inconsistent_bags = bag_label_counts[bag_label_counts > 1]
    if len(inconsistent_bags):
        raise ValueError(
            "Each PID must have one bag label. Inconsistent PIDs: "
            f"{inconsistent_bags.index.tolist()[:10]}"
        )

    X_df = df_sorted[features].apply(pd.to_numeric, errors="coerce")
    X = X_df.fillna(feature_fill_value).to_numpy(dtype=np.float64)
    y = grouped[BAG_LABEL_COL].first().astype(float).to_numpy(dtype=np.float64)
    group_sizes = grouped.size().to_numpy(dtype=int)

    instance_y = pd.to_numeric(
        df_sorted[INST_LABEL_COL],
        errors="coerce",
    ).to_numpy(dtype=np.float64)


    instance_names = (
        df_sorted[BAG_ID_COL].astype(str)
        + "_embryo"
        + df_sorted[INST_ID_COL].astype(str)
    ).to_numpy()

    return X, y, group_sizes, instance_y, instance_names, df_sorted



def make_model():
    return GradBoostingClassifier(
        lam_2=0.001,
        lr=1.0,
        max_depth=2,
        splitter="random",
        n_estimators=100,
        n_update_iterations=1,
        embedding_size=8,
        nn_lr=1e-3,
        nn_num_heads=1,
        nn_steps=10,
        dropout=0.0,
        # Pairwise RankNet attention-logit ranking: known positive instances
        # should receive higher raw attention logits than known negative
        # instances in the same bag.
        # lambda_rank=0.5,
        lambda_rank=0,
        # rank_margin=0.0,
        # Optional pointwise supervision on the same raw attention logits.
        # lambda_inst=0.1,
        lambda_inst=0.0,
    )


def split_by_bag(flat_values, group_sizes):
    starts = np.r_[0, np.cumsum(group_sizes[:-1])]
    return [
        np.asarray(flat_values[start:start + size])
        for start, size in zip(starts, group_sizes)
    ]


def ranked_instances_for_bag(bag_id, names, labels, logits, weights):
    order = np.argsort(-logits)
    return [
        {
            "bag_id": bag_id,
            "rank": rank,
            "instance": names[idx],
            "instance_label": labels[idx],
            "attention_logit": logits[idx],
            "attention_weight": weights[idx],
        }
        for rank, idx in enumerate(order, start=1)
    ]


def generate_rankings(group_sizes, instance_names, instance_y, logits, weights):
    """Return one row per embryo, ranked within each bag by attention logit."""
    names_by_bag = split_by_bag(instance_names, group_sizes)
    labels_by_bag = split_by_bag(instance_y, group_sizes)

    rows = []
    for bag_id, (names, labels, bag_logits, bag_weights) in enumerate(
        zip(names_by_bag, labels_by_bag, logits, weights)
    ):
        rows.extend(
            ranked_instances_for_bag(bag_id, names, labels, bag_logits, bag_weights)
        )

    return pd.DataFrame(rows)


def print_rankings(group_sizes, instance_names, instance_y, logits, weights):
    rankings = generate_rankings(group_sizes, instance_names, instance_y, logits, weights)

    for bag_id, bag_rankings in rankings.groupby("bag_id", sort=False):
        print(f"\nBag {bag_id}")
        print("rank  instance       label  logit      weight")
        for row in bag_rankings.to_dict("records"):
            label = "?" if np.isnan(row["instance_label"]) else row["instance_label"]
            print(
                f"{row['rank']:>4}  "
                f"{row['instance']:<13} "
                f"{label!s:>5}  "
                f"{row['attention_logit']:>8.4f}  "
                f"{row['attention_weight']:>8.4f}"
            )


def main():
    # For your dataframe, replace make_small_mil_dataset() with:
    #
    # X, y, group_sizes, instance_y, instance_names, _ = make_mil_arrays_from_dataframe(
    #     df_train,
    #     include_transfer_order=False,
    # )
    #
    # Keep include_transfer_order=False unless transfer order is available at
    # prediction time and you are comfortable using it as an embryo feature.
    # X, y, group_sizes, instance_y, instance_names = make_small_mil_dataset()
    import os
    import pickle
    import cloudpickle
    path="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/multiinstanceEmbryoRanking/analysis"
    df_train=pd.read_csv(os.path.join(path, "data/train_corrected_remove.csv"))
    df_test=pd.read_csv(os.path.join(path, "data/test_corrected_remove.csv"))
    X, y, group_sizes, instance_y, instance_names, _ = make_mil_arrays_from_dataframe(df_train)
    X_te, y_te, group_sizes_te, instance_y_te, instance_names_te, df_test_sorted = make_mil_arrays_from_dataframe(df_test)
    print(
        "Bag counts:",
        f"train df={df_train[BAG_ID_COL].nunique(dropna=False)} converted={len(group_sizes)}",
        f"test df={df_test[BAG_ID_COL].nunique(dropna=False)} converted={len(group_sizes_te)}",
    )

    model = make_model()
    print(
        "Training with: bag BCE + "
        f"{model.params['lambda_rank']} * instance RankNet + "
        f"{model.params['lambda_inst']} * instance BCE"
    )
    print(
        "Train arrays:",
        f"X={type(X).__name__}{X.shape}/{X.dtype}",
        f"y={type(y).__name__}{y.shape}/{y.dtype}",
        f"group_sizes={type(group_sizes).__name__}{group_sizes.shape}/{group_sizes.dtype}",
        f"instance_y={type(instance_y).__name__}{instance_y.shape}/{instance_y.dtype}",
    )
    model.fit(X, y, group_sizes, instance_y=instance_y)

    # Raw logits are the supervised scores used by RankNet and instance BCE.
    # Weights are the softmax-normalized attention scores used by aggregation.
    attention_logits = model.predict_attention_logits(X_te, group_sizes_te, grouped=True)
    attention_weights = model.predict_attention_weights(X_te, group_sizes_te, grouped=True)

    bag_probs = model.predict_proba(X_te, group_sizes_te)
    bag_preds = np.argmax(bag_probs, axis=1)

    rankings = generate_rankings(
        group_sizes_te,
        instance_names_te,
        instance_y_te,
        attention_logits,
        attention_weights,
    )
    rankings[BAG_ID_COL] = df_test_sorted[BAG_ID_COL].to_numpy()
    rankings[INST_ID_COL] = df_test_sorted[INST_ID_COL].to_numpy()
    rankings[BAG_LABEL_COL] = df_test_sorted[BAG_LABEL_COL].to_numpy()
    rankings["bag_prediction"] = np.repeat(bag_preds, group_sizes_te)
    # rankings["bag_probability_0"] = np.repeat(bag_probs[:, 0], group_sizes_te)
    rankings["bag_probability_1"] = np.repeat(bag_probs[:, 1], group_sizes_te)
    rankings["bag_livebirth_probability"] = rankings["bag_probability_1"]
    rankings.to_csv(os.path.join(path, "test_attention_rankings_corrected_norankingloss_day56.csv"), index=False)
    # model.save(os.path.join(path, "rank_attention_mil_model_corrected.pkl"))
    # pickle save model 
    with open(os.path.join(path, "rank_attention_mil_model_corrected_remove_norankingloss_day56.pkl"), 'wb') as file:
        cloudpickle.dump(model, file)
    print(rankings.head(20))


if __name__ == "__main__":
    main()
