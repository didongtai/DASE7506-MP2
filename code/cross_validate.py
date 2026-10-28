"""Five-fold training-only check of answer grouping and late fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs
from solution_utils import (
    build_feature_blocks,
    choices_from_answer_groups,
    choices_from_scores,
    normalize_features,
    select_matrix,
)
from train import (
    ARITHMETIC_ALPHA,
    ARITHMETIC_COLUMNS,
    COUNT_BONUSES,
    FUSION_EMBEDDING_ALPHA,
    FUSION_GAMMAS,
    FUSION_SURFACE_ALPHA,
    GROUP_TEMPERATURES,
    RANDOM_FEATURE_ALPHA,
    RANDOM_FEATURE_DIM,
    RANDOM_FEATURE_SCALE,
    RANDOM_FEATURE_SEED,
    ENHANCED_ARITHMETIC_ALPHA,
    NUMERIC_ALPHA,
    V4_ARITHMETIC_GAMMA,
    V4_COUNT_BONUS,
    V4_NONLINEAR_GAMMA,
    V4_SURFACE_GAMMA,
    V5_ARITHMETIC_GAMMA,
    V5_COUNT_BONUS,
    V5_ENHANCED_ARITHMETIC_GAMMA,
    V5_NONLINEAR_GAMMA,
    V5_NUMERIC_GAMMA,
    V6_COUNT_BONUS,
    V6_NUMERIC_LOGISTIC_GAMMA,
    V6_NUMERIC_LOGISTIC_L2,
    V6_NUMERIC_RIDGE_GAMMA,
    evaluate,
    fit_logistic,
    fit_ridge,
    labels_for,
)


def fold_for(identifier, folds, salt=""):
    digest = hashlib.sha256((salt + identifier).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-salt", default="")
    parser.add_argument("--output", type=Path, default=Path("experiments/cross-validation.json"))
    args = parser.parse_args()
    started = time.perf_counter()

    bank, arrays = load_inputs("train", args.data_dir)
    labels = labels_for("train", bank, args.data_dir)
    blocks = build_feature_blocks(bank, arrays["embeddings"])
    assignments = np.asarray(
        [fold_for(row["id"], args.folds, args.fold_salt) for row in bank]
    )
    out_of_fold_scores = {
        "embedding": np.zeros((len(bank), 4), dtype=np.float32),
        "surface": np.zeros((len(bank), 4), dtype=np.float32),
        "v4_base": np.zeros((len(bank), 4), dtype=np.float32),
        "nonlinear": np.zeros((len(bank), 4), dtype=np.float32),
        "arithmetic": np.zeros((len(bank), 4), dtype=np.float32),
        "enhanced_arithmetic": np.zeros((len(bank), 4), dtype=np.float32),
        "numeric": np.zeros((len(bank), 4), dtype=np.float32),
        "numeric_logistic": np.zeros((len(bank), 4), dtype=np.float32),
    }
    matrices = {
        feature_set: select_matrix(blocks, feature_set)
        for feature_set in ("embedding", "surface")
    }
    random_projection = np.random.default_rng(RANDOM_FEATURE_SEED).normal(
        size=(blocks["embedding"].shape[2], RANDOM_FEATURE_DIM)
    ).astype(np.float32)
    random_matrix = np.tanh(
        RANDOM_FEATURE_SCALE
        * (blocks["embedding"].reshape(-1, 1024) @ random_projection)
    )
    arithmetic_matrix = select_matrix(blocks, "arithmetic")[:, ARITHMETIC_COLUMNS]
    enhanced_arithmetic_matrix = select_matrix(blocks, "arithmetic_v2")
    numeric_matrix = select_matrix(blocks, "numeric")

    fold_sizes = []
    for fold in range(args.folds):
        held_out_questions = assignments == fold
        train_rows = np.repeat(~held_out_questions, 4)
        held_out_rows = np.repeat(held_out_questions, 4)
        fold_scores = {}
        fold_train_scores = {}
        for feature_set, alpha in (
            ("embedding", FUSION_EMBEDDING_ALPHA),
            ("surface", FUSION_SURFACE_ALPHA),
        ):
            matrix = matrices[feature_set]
            train_x, mean, scale = normalize_features(matrix[train_rows])
            held_out_x, _, _ = normalize_features(matrix[held_out_rows], mean, scale)
            weights = fit_ridge(train_x, labels[~held_out_questions], alpha=alpha)
            fold_train_scores[feature_set] = train_x @ weights
            fold_scores[feature_set] = held_out_x @ weights
            out_of_fold_scores[feature_set][held_out_questions] = fold_scores[
                feature_set
            ].reshape(-1, 4)

        train_base = fold_train_scores["embedding"] + V4_SURFACE_GAMMA * fold_train_scores[
            "surface"
        ]
        held_out_base = fold_scores["embedding"] + V4_SURFACE_GAMMA * fold_scores[
            "surface"
        ]
        out_of_fold_scores["v4_base"][held_out_questions] = (
            held_out_base / max(float(train_base.std()), 1e-8)
        ).reshape(-1, 4)

        for name, matrix, alpha in (
            ("nonlinear", random_matrix, RANDOM_FEATURE_ALPHA),
            ("arithmetic", arithmetic_matrix, ARITHMETIC_ALPHA),
            ("enhanced_arithmetic", enhanced_arithmetic_matrix, ENHANCED_ARITHMETIC_ALPHA),
            ("numeric", numeric_matrix, NUMERIC_ALPHA),
        ):
            train_x, mean, scale = normalize_features(matrix[train_rows])
            held_out_x, _, _ = normalize_features(matrix[held_out_rows], mean, scale)
            weights = fit_ridge(train_x, labels[~held_out_questions], alpha=alpha)
            train_scores = train_x @ weights
            out_of_fold_scores[name][held_out_questions] = (
                (held_out_x @ weights) / max(float(train_scores.std()), 1e-8)
            ).reshape(-1, 4)
        train_numeric_x, numeric_mean, numeric_scale = normalize_features(
            numeric_matrix[train_rows]
        )
        held_numeric_x, _, _ = normalize_features(
            numeric_matrix[held_out_rows], numeric_mean, numeric_scale
        )
        numeric_logistic_weights = fit_logistic(
            train_numeric_x, labels[~held_out_questions], V6_NUMERIC_LOGISTIC_L2
        )
        numeric_logistic_train_scores = train_numeric_x @ numeric_logistic_weights
        out_of_fold_scores["numeric_logistic"][held_out_questions] = (
            (held_numeric_x @ numeric_logistic_weights)
            / max(float(numeric_logistic_train_scores.std()), 1e-8)
        ).reshape(-1, 4)
        fold_sizes.append(int(held_out_questions.sum()))

    results = []
    candidate_choices = choices_from_scores(out_of_fold_scores["embedding"], blocks["valid"])
    results.append(
        {
            "method": "candidate_max",
            **evaluate("train", bank, candidate_choices, args.data_dir),
        }
    )
    v6_scores = (
        out_of_fold_scores["v4_base"]
        + V5_NONLINEAR_GAMMA * out_of_fold_scores["nonlinear"]
        + V5_ARITHMETIC_GAMMA * out_of_fold_scores["arithmetic"]
        + V5_ENHANCED_ARITHMETIC_GAMMA
        * out_of_fold_scores["enhanced_arithmetic"]
        + V6_NUMERIC_RIDGE_GAMMA * out_of_fold_scores["numeric"]
        + V6_NUMERIC_LOGISTIC_GAMMA * out_of_fold_scores["numeric_logistic"]
    )
    v6_choices = choices_from_answer_groups(
        bank, v6_scores, reducer="max", count_bonus=V6_COUNT_BONUS
    )
    results.append(
        {
            "method": "structured_logistic_numeric_fusion",
            "surface_gamma": V4_SURFACE_GAMMA,
            "nonlinear_gamma": V5_NONLINEAR_GAMMA,
            "arithmetic_gamma": V5_ARITHMETIC_GAMMA,
            "enhanced_arithmetic_gamma": V5_ENHANCED_ARITHMETIC_GAMMA,
            "numeric_ridge_gamma": V6_NUMERIC_RIDGE_GAMMA,
            "numeric_logistic_l2": V6_NUMERIC_LOGISTIC_L2,
            "numeric_logistic_gamma": V6_NUMERIC_LOGISTIC_GAMMA,
            "count_bonus": V6_COUNT_BONUS,
            **evaluate("train", bank, v6_choices, args.data_dir),
        }
    )

    v5_scores = (
        out_of_fold_scores["v4_base"]
        + V5_NONLINEAR_GAMMA * out_of_fold_scores["nonlinear"]
        + V5_ARITHMETIC_GAMMA * out_of_fold_scores["arithmetic"]
        + V5_ENHANCED_ARITHMETIC_GAMMA
        * out_of_fold_scores["enhanced_arithmetic"]
        + V5_NUMERIC_GAMMA * out_of_fold_scores["numeric"]
    )
    v5_choices = choices_from_answer_groups(
        bank, v5_scores, reducer="max", count_bonus=V5_COUNT_BONUS
    )
    results.append(
        {
            "method": "structured_numeric_arithmetic_fusion",
            "surface_gamma": V4_SURFACE_GAMMA,
            "nonlinear_gamma": V5_NONLINEAR_GAMMA,
            "arithmetic_gamma": V5_ARITHMETIC_GAMMA,
            "enhanced_arithmetic_gamma": V5_ENHANCED_ARITHMETIC_GAMMA,
            "numeric_gamma": V5_NUMERIC_GAMMA,
            "count_bonus": V5_COUNT_BONUS,
            **evaluate("train", bank, v5_choices, args.data_dir),
        }
    )
    for temperature in GROUP_TEMPERATURES:
        choices = choices_from_answer_groups(
            bank, out_of_fold_scores["embedding"], temperature=temperature
        )
        results.append(
            {
                "method": "answer_group_logsumexp",
                "temperature": temperature,
                **evaluate("train", bank, choices, args.data_dir),
            }
        )

    for gamma in FUSION_GAMMAS:
        fused_scores = (
            out_of_fold_scores["embedding"] + gamma * out_of_fold_scores["surface"]
        )
        for count_bonus in COUNT_BONUSES:
            choices = choices_from_answer_groups(
                bank, fused_scores, reducer="max", count_bonus=count_bonus
            )
            results.append(
                {
                    "method": "late_fusion_max_count",
                    "fusion_gamma": gamma,
                    "count_bonus": count_bonus,
                    **evaluate("train", bank, choices, args.data_dir),
                }
            )

    v4_scores = (
        out_of_fold_scores["v4_base"]
        + V4_NONLINEAR_GAMMA * out_of_fold_scores["nonlinear"]
        + V4_ARITHMETIC_GAMMA * out_of_fold_scores["arithmetic"]
    )
    v4_choices = choices_from_answer_groups(
        bank, v4_scores, reducer="max", count_bonus=V4_COUNT_BONUS
    )
    results.append(
        {
            "method": "nonlinear_arithmetic_fusion_max_count",
            "surface_gamma": V4_SURFACE_GAMMA,
            "nonlinear_gamma": V4_NONLINEAR_GAMMA,
            "arithmetic_gamma": V4_ARITHMETIC_GAMMA,
            "count_bonus": V4_COUNT_BONUS,
            **evaluate("train", bank, v4_choices, args.data_dir),
        }
    )

    record = {
        "split": "train",
        "test_used": False,
        "fold_assignment": "sha256(question_id) modulo folds",
        "fold_salt": args.fold_salt,
        "folds": args.folds,
        "fold_sizes": fold_sizes,
        "embedding_ridge_alpha": FUSION_EMBEDDING_ALPHA,
        "surface_ridge_alpha": FUSION_SURFACE_ALPHA,
        "fusion_gamma_grid": list(FUSION_GAMMAS),
        "count_bonus_grid": list(COUNT_BONUSES),
        "v4_fixed_configuration": {
            "random_feature_seed": RANDOM_FEATURE_SEED,
            "random_feature_dim": RANDOM_FEATURE_DIM,
            "random_feature_scale": RANDOM_FEATURE_SCALE,
            "random_feature_alpha": RANDOM_FEATURE_ALPHA,
            "arithmetic_columns": list(ARITHMETIC_COLUMNS),
            "arithmetic_alpha": ARITHMETIC_ALPHA,
            "surface_gamma": V4_SURFACE_GAMMA,
            "nonlinear_gamma": V4_NONLINEAR_GAMMA,
            "arithmetic_gamma": V4_ARITHMETIC_GAMMA,
            "count_bonus": V4_COUNT_BONUS,
        },
        "v5_fixed_configuration": {
            "enhanced_arithmetic_alpha": ENHANCED_ARITHMETIC_ALPHA,
            "numeric_alpha": NUMERIC_ALPHA,
            "surface_gamma": V4_SURFACE_GAMMA,
            "nonlinear_gamma": V5_NONLINEAR_GAMMA,
            "arithmetic_gamma": V5_ARITHMETIC_GAMMA,
            "enhanced_arithmetic_gamma": V5_ENHANCED_ARITHMETIC_GAMMA,
            "numeric_gamma": V5_NUMERIC_GAMMA,
            "count_bonus": V5_COUNT_BONUS,
        },
        "v6_fixed_configuration": {
            "numeric_ridge_gamma": V6_NUMERIC_RIDGE_GAMMA,
            "numeric_logistic_l2": V6_NUMERIC_LOGISTIC_L2,
            "numeric_logistic_gamma": V6_NUMERIC_LOGISTIC_GAMMA,
            "count_bonus": V6_COUNT_BONUS,
        },
        "seconds": time.perf_counter() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
