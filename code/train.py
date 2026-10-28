"""Train and select the lightweight MP2 verifier using train/validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl, score_choices
from solution_utils import (
    FEATURE_VERSION,
    build_feature_blocks,
    choices_from_answer_groups,
    choices_from_scores,
    majority_choice,
    normalize_features,
    select_matrix,
)


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
FEATURE_SETS = ("surface", "embedding", "combined")
GROUP_TEMPERATURES = (0.1, 0.25, 0.5, 1.0)
FUSION_GAMMAS = (0.05, 0.1, 0.15, 0.2, 0.25)
COUNT_BONUSES = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15)
FUSION_EMBEDDING_ALPHA = 0.01
FUSION_SURFACE_ALPHA = 100.0
RANDOM_FEATURE_SEED = 1
RANDOM_FEATURE_DIM = 128
RANDOM_FEATURE_SCALE = 0.5
RANDOM_FEATURE_ALPHA = 1000.0
ARITHMETIC_COLUMNS = (2, 4, 5, 6)
ARITHMETIC_ALPHA = 100.0
V4_SURFACE_GAMMA = 0.25
V4_NONLINEAR_GAMMA = 0.3
V4_ARITHMETIC_GAMMA = 0.2
V4_COUNT_BONUS = 0.2
ENHANCED_ARITHMETIC_ALPHA = 100.0
NUMERIC_ALPHA = 1.0
V5_NONLINEAR_GAMMA = 0.475
V5_ARITHMETIC_GAMMA = 0.1
V5_ENHANCED_ARITHMETIC_GAMMA = 0.25
V5_NUMERIC_GAMMA = 0.75
V5_COUNT_BONUS = 0.275
V6_NUMERIC_RIDGE_GAMMA = 0.5
V6_NUMERIC_LOGISTIC_L2 = 100.0
V6_NUMERIC_LOGISTIC_GAMMA = 0.9
V6_COUNT_BONUS = 0.275
LISTWISE_L2 = (0.0, 0.0001, 0.001)
LISTWISE_STEPS = 150
LISTWISE_LEARNING_RATE = 0.02


def labels_for(split, bank, data_dir):
    references = read_jsonl(data_dir / f"gsm8k_{split}.jsonl")
    if len(references) != len(bank):
        raise ValueError("Reference length mismatch")
    labels = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, (row, reference) in enumerate(zip(bank, references)):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Reference alignment mismatch")
        gold = parse_answer(reference["answer"])
        for candidate_index, candidate in enumerate(row["candidates"]):
            labels[row_index, candidate_index] = parse_answer(candidate["text"]) == gold
    return labels


def fit_ridge(matrix, labels, alpha):
    """Fit a small deterministic linear verifier with L2 regularization."""
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    gram = x.T @ x
    gram.flat[:: len(gram) + 1] += alpha
    return np.linalg.solve(gram, x.T @ y).astype(np.float32)


def fit_logistic(matrix, labels, l2, steps=15):
    """Fit a deterministic logistic verifier with Newton updates."""
    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    identity = np.eye(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        logits = np.clip(x @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        variance = np.maximum(probability * (1.0 - probability), 1e-5)
        gradient = x.T @ (probability - y) + l2 * weights
        hessian = (x.T * variance) @ x + l2 * identity
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if np.linalg.norm(update) < 1e-6:
            break
    return weights.astype(np.float32)


def fit_group_softmax(matrix, labels, initial_weights, l2, steps=LISTWISE_STEPS):
    """Optimize likelihood of the correct parsed-answer group.

    When several candidates share the gold answer, their softmax mass is added.
    Questions where every candidate is correct therefore contribute zero loss.
    """
    x = np.asarray(matrix, dtype=np.float32)
    correct = np.asarray(labels, dtype=bool).reshape(-1, 4)
    if np.any(~correct.any(axis=1)):
        raise ValueError("Every training question must have a correct candidate")
    weights = np.asarray(initial_weights, dtype=np.float32).copy()
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    n = len(correct)
    loss = 0.0

    for step in range(1, steps + 1):
        scores = (x @ weights).reshape(n, 4)
        peak = scores.max(axis=1, keepdims=True)
        exp_all = np.exp(scores - peak)
        probabilities = exp_all / exp_all.sum(axis=1, keepdims=True)

        correct_scores = np.where(correct, scores, -np.inf)
        correct_peak = correct_scores.max(axis=1, keepdims=True)
        exp_correct = np.where(correct, np.exp(correct_scores - correct_peak), 0.0)
        correct_probabilities = exp_correct / exp_correct.sum(axis=1, keepdims=True)

        score_gradient = (probabilities - correct_probabilities) / n
        gradient = x.T @ score_gradient.reshape(-1) + l2 * weights
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        weights -= LISTWISE_LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)

        if step == steps:
            log_all = peak[:, 0] + np.log(exp_all.sum(axis=1))
            log_correct = correct_peak[:, 0] + np.log(exp_correct.sum(axis=1))
            loss = float(np.mean(log_all - log_correct) + 0.5 * l2 * np.dot(weights, weights))
    return weights, loss


def evaluate(split, bank, choices, data_dir):
    ids = [row["id"] for row in bank]
    result = score_choices(split, ids, choices, data_dir)
    answers = [[parse_answer(candidate["text"]) for candidate in row["candidates"]] for row in bank]
    disagreement = [len(set(values)) > 1 for values in answers]
    references = read_jsonl(data_dir / f"gsm8k_{split}.jsonl")
    hard_correct = sum(
        flag and answers[i][choices[i]] == parse_answer(references[i]["answer"])
        for i, flag in enumerate(disagreement)
    )
    hard_n = sum(disagreement)
    return {
        **result,
        "disagreement_correct": hard_correct,
        "disagreement_n": hard_n,
        "disagreement_accuracy": 100.0 * hard_correct / hard_n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/model_bundle"))
    args = parser.parse_args()
    started = time.perf_counter()

    train_bank, train_arrays = load_inputs("train", args.data_dir)
    validation_bank, validation_arrays = load_inputs("validation", args.data_dir)
    train_labels = labels_for("train", train_bank, args.data_dir)
    train_blocks = build_feature_blocks(train_bank, train_arrays["embeddings"])
    validation_blocks = build_feature_blocks(validation_bank, validation_arrays["embeddings"])

    results = []
    for name, choices in (
        ("first", [0] * len(validation_bank)),
        ("majority", [majority_choice(row) for row in validation_bank]),
    ):
        results.append({"method": name, **evaluate("validation", validation_bank, choices, args.data_dir)})

    candidates = []
    prepared = {}
    ridge_models = {}
    for feature_set in FEATURE_SETS:
        train_x = select_matrix(train_blocks, feature_set)
        validation_x = select_matrix(validation_blocks, feature_set)
        train_x, mean, scale = normalize_features(train_x)
        validation_x, _, _ = normalize_features(validation_x, mean, scale)
        prepared[feature_set] = (train_x, validation_x, mean, scale)
        for alpha in ALPHAS:
            weights = fit_ridge(train_x, train_labels, alpha)
            ridge_models[(feature_set, alpha)] = (weights, mean, scale)
            scores = (validation_x @ weights).reshape(len(validation_bank), 4)
            choices = choices_from_scores(scores, validation_blocks["valid"])
            metrics = evaluate("validation", validation_bank, choices, args.data_dir)
            record = {"method": feature_set, "alpha": alpha, **metrics}
            results.append(record)
            candidates.append(
                (
                    record,
                    {
                        "feature_set": feature_set,
                        "weights": weights,
                        "mean": mean,
                        "scale": scale,
                        "selection_mode": "candidate",
                        "temperature": 1.0,
                    },
                )
            )

            if feature_set in ("embedding", "combined"):
                for temperature in GROUP_TEMPERATURES:
                    group_choices = choices_from_answer_groups(
                        validation_bank, scores, temperature=temperature
                    )
                    group_metrics = evaluate(
                        "validation", validation_bank, group_choices, args.data_dir
                    )
                    group_record = {
                        "method": f"{feature_set}_ridge_group",
                        "alpha": alpha,
                        "temperature": temperature,
                        **group_metrics,
                    }
                    results.append(group_record)
                    candidates.append(
                        (
                            group_record,
                            {
                                "feature_set": feature_set,
                                "weights": weights,
                                "mean": mean,
                                "scale": scale,
                                "selection_mode": "answer_group",
                                "temperature": temperature,
                            },
                        )
                    )

    # Late fusion preserves the strong relative-embedding verifier while using
    # the independently regularized surface model as a small correction.  The
    # logarithmic count term rewards corroborating traces without tying the
    # answer-group strength to the numerical scale of model scores.
    embedding_weights, embedding_mean, embedding_scale = ridge_models[
        ("embedding", FUSION_EMBEDDING_ALPHA)
    ]
    surface_weights, surface_mean, surface_scale = ridge_models[
        ("surface", FUSION_SURFACE_ALPHA)
    ]
    embedding_scores = (
        prepared["embedding"][1] @ embedding_weights
    ).reshape(len(validation_bank), 4)
    surface_scores = (
        prepared["surface"][1] @ surface_weights
    ).reshape(len(validation_bank), 4)
    for gamma in FUSION_GAMMAS:
        fused_scores = embedding_scores + gamma * surface_scores
        for count_bonus in COUNT_BONUSES:
            choices = choices_from_answer_groups(
                validation_bank,
                fused_scores,
                reducer="max",
                count_bonus=count_bonus,
            )
            metrics = evaluate("validation", validation_bank, choices, args.data_dir)
            record = {
                "method": "late_fusion_max_count",
                "embedding_alpha": FUSION_EMBEDDING_ALPHA,
                "surface_alpha": FUSION_SURFACE_ALPHA,
                "fusion_gamma": gamma,
                "count_bonus": count_bonus,
                **metrics,
            }
            results.append(record)
            candidates.append(
                (
                    record,
                    {
                        "model_kind": "late_fusion",
                        "fusion_gamma": gamma,
                        "count_bonus": count_bonus,
                        "embedding_weights": embedding_weights,
                        "embedding_mean": embedding_mean,
                        "embedding_scale": embedding_scale,
                        "surface_weights": surface_weights,
                        "surface_mean": surface_mean,
                        "surface_scale": surface_scale,
                    },
                )
            )

    # V4 adds two compact, independently regularized checks. A fixed random
    # projection followed by tanh supplies low-rank nonlinear interactions in
    # the relative embedding. The arithmetic model uses only four features
    # describing explicitly inconsistent equation chains. Each component score
    # is scaled using training predictions before the fixed late fusion.
    train_embedding_x, validation_embedding_x = prepared["embedding"][:2]
    train_surface_x, validation_surface_x = prepared["surface"][:2]
    train_embedding_scores = train_embedding_x @ embedding_weights
    train_surface_scores = train_surface_x @ surface_weights
    train_base_scores = (
        train_embedding_scores + V4_SURFACE_GAMMA * train_surface_scores
    )
    validation_base_scores = (
        validation_embedding_x @ embedding_weights
        + V4_SURFACE_GAMMA * (validation_surface_x @ surface_weights)
    )
    base_score_scale = float(train_base_scores.std())

    random_projection = np.random.default_rng(RANDOM_FEATURE_SEED).normal(
        size=(train_blocks["embedding"].shape[2], RANDOM_FEATURE_DIM)
    ).astype(np.float32)
    train_random_x = np.tanh(
        RANDOM_FEATURE_SCALE
        * (train_blocks["embedding"].reshape(-1, 1024) @ random_projection)
    )
    validation_random_x = np.tanh(
        RANDOM_FEATURE_SCALE
        * (validation_blocks["embedding"].reshape(-1, 1024) @ random_projection)
    )
    train_random_x, random_mean, random_scale = normalize_features(train_random_x)
    validation_random_x, _, _ = normalize_features(
        validation_random_x, random_mean, random_scale
    )
    random_weights = fit_ridge(train_random_x, train_labels, RANDOM_FEATURE_ALPHA)
    train_random_scores = train_random_x @ random_weights
    random_score_scale = float(train_random_scores.std())

    train_arithmetic_x = select_matrix(train_blocks, "arithmetic")[:, ARITHMETIC_COLUMNS]
    validation_arithmetic_x = select_matrix(validation_blocks, "arithmetic")[
        :, ARITHMETIC_COLUMNS
    ]
    train_arithmetic_x, arithmetic_mean, arithmetic_scale = normalize_features(
        train_arithmetic_x
    )
    validation_arithmetic_x, _, _ = normalize_features(
        validation_arithmetic_x, arithmetic_mean, arithmetic_scale
    )
    arithmetic_weights = fit_ridge(
        train_arithmetic_x, train_labels, ARITHMETIC_ALPHA
    )
    train_arithmetic_scores = train_arithmetic_x @ arithmetic_weights
    arithmetic_score_scale = float(train_arithmetic_scores.std())

    nonlinear_validation_scores = (
        validation_random_x @ random_weights / random_score_scale
    ).reshape(len(validation_bank), 4)
    v4_without_arithmetic = (
        validation_base_scores / base_score_scale
    ).reshape(len(validation_bank), 4) + V4_NONLINEAR_GAMMA * nonlinear_validation_scores
    for method, fused_scores, arithmetic_gamma in (
        ("nonlinear_fusion_max_count", v4_without_arithmetic, 0.0),
        (
            "nonlinear_arithmetic_fusion_max_count",
            v4_without_arithmetic
            + V4_ARITHMETIC_GAMMA
            * (
                validation_arithmetic_x @ arithmetic_weights / arithmetic_score_scale
            ).reshape(len(validation_bank), 4),
            V4_ARITHMETIC_GAMMA,
        ),
    ):
        choices = choices_from_answer_groups(
            validation_bank,
            fused_scores,
            reducer="max",
            count_bonus=V4_COUNT_BONUS,
        )
        metrics = evaluate("validation", validation_bank, choices, args.data_dir)
        record = {
            "method": method,
            "surface_gamma": V4_SURFACE_GAMMA,
            "nonlinear_gamma": V4_NONLINEAR_GAMMA,
            "arithmetic_gamma": arithmetic_gamma,
            "count_bonus": V4_COUNT_BONUS,
            **metrics,
        }
        results.append(record)
        if arithmetic_gamma:
            candidates.append(
                (
                    record,
                    {
                        "model_kind": "nonlinear_arithmetic_fusion",
                        "embedding_weights": embedding_weights,
                        "embedding_mean": embedding_mean,
                        "embedding_scale": embedding_scale,
                        "surface_weights": surface_weights,
                        "surface_mean": surface_mean,
                        "surface_scale": surface_scale,
                        "base_score_scale": base_score_scale,
                        "random_projection": random_projection,
                        "random_weights": random_weights,
                        "random_mean": random_mean,
                        "random_scale": random_scale,
                        "random_score_scale": random_score_scale,
                        "arithmetic_weights": arithmetic_weights,
                        "arithmetic_mean": arithmetic_mean,
                        "arithmetic_scale": arithmetic_scale,
                        "arithmetic_score_scale": arithmetic_score_scale,
                    },
                )
            )

    # V5 adds two small structured checks selected with three independent
    # training-only fold assignments. The tolerant arithmetic parser accepts
    # ordinary unit suffixes, while the numeric model describes an answer's
    # rank among its peers and simple reachability from question numbers.
    train_enhanced_arithmetic_x = select_matrix(train_blocks, "arithmetic_v2")
    validation_enhanced_arithmetic_x = select_matrix(
        validation_blocks, "arithmetic_v2"
    )
    train_enhanced_arithmetic_x, enhanced_arithmetic_mean, enhanced_arithmetic_scale = normalize_features(
        train_enhanced_arithmetic_x
    )
    validation_enhanced_arithmetic_x, _, _ = normalize_features(
        validation_enhanced_arithmetic_x,
        enhanced_arithmetic_mean,
        enhanced_arithmetic_scale,
    )
    enhanced_arithmetic_weights = fit_ridge(
        train_enhanced_arithmetic_x, train_labels, ENHANCED_ARITHMETIC_ALPHA
    )
    train_enhanced_arithmetic_scores = (
        train_enhanced_arithmetic_x @ enhanced_arithmetic_weights
    )
    enhanced_arithmetic_score_scale = float(
        train_enhanced_arithmetic_scores.std()
    )

    train_numeric_x = select_matrix(train_blocks, "numeric")
    validation_numeric_x = select_matrix(validation_blocks, "numeric")
    train_numeric_x, numeric_mean, numeric_scale = normalize_features(train_numeric_x)
    validation_numeric_x, _, _ = normalize_features(
        validation_numeric_x, numeric_mean, numeric_scale
    )
    numeric_weights = fit_ridge(train_numeric_x, train_labels, NUMERIC_ALPHA)
    train_numeric_scores = train_numeric_x @ numeric_weights
    numeric_score_scale = float(train_numeric_scores.std())

    v5_scores = (
        (validation_base_scores / base_score_scale).reshape(len(validation_bank), 4)
        + V5_NONLINEAR_GAMMA * nonlinear_validation_scores
        + V5_ARITHMETIC_GAMMA
        * (validation_arithmetic_x @ arithmetic_weights / arithmetic_score_scale).reshape(
            len(validation_bank), 4
        )
        + V5_ENHANCED_ARITHMETIC_GAMMA
        * (
            validation_enhanced_arithmetic_x
            @ enhanced_arithmetic_weights
            / enhanced_arithmetic_score_scale
        ).reshape(len(validation_bank), 4)
        + V5_NUMERIC_GAMMA
        * (validation_numeric_x @ numeric_weights / numeric_score_scale).reshape(
            len(validation_bank), 4
        )
    )
    v5_choices = choices_from_answer_groups(
        validation_bank,
        v5_scores,
        reducer="max",
        count_bonus=V5_COUNT_BONUS,
    )
    v5_metrics = evaluate("validation", validation_bank, v5_choices, args.data_dir)
    v5_record = {
        "method": "structured_numeric_arithmetic_fusion",
        "surface_gamma": V4_SURFACE_GAMMA,
        "nonlinear_gamma": V5_NONLINEAR_GAMMA,
        "arithmetic_gamma": V5_ARITHMETIC_GAMMA,
        "enhanced_arithmetic_gamma": V5_ENHANCED_ARITHMETIC_GAMMA,
        "numeric_gamma": V5_NUMERIC_GAMMA,
        "count_bonus": V5_COUNT_BONUS,
        **v5_metrics,
    }
    results.append(v5_record)
    candidates.append(
        (
            v5_record,
            {
                "model_kind": "structured_fusion",
                "embedding_weights": embedding_weights,
                "embedding_mean": embedding_mean,
                "embedding_scale": embedding_scale,
                "surface_weights": surface_weights,
                "surface_mean": surface_mean,
                "surface_scale": surface_scale,
                "base_score_scale": base_score_scale,
                "random_projection": random_projection,
                "random_weights": random_weights,
                "random_mean": random_mean,
                "random_scale": random_scale,
                "random_score_scale": random_score_scale,
                "arithmetic_weights": arithmetic_weights,
                "arithmetic_mean": arithmetic_mean,
                "arithmetic_scale": arithmetic_scale,
                "arithmetic_score_scale": arithmetic_score_scale,
                "enhanced_arithmetic_weights": enhanced_arithmetic_weights,
                "enhanced_arithmetic_mean": enhanced_arithmetic_mean,
                "enhanced_arithmetic_scale": enhanced_arithmetic_scale,
                "enhanced_arithmetic_score_scale": enhanced_arithmetic_score_scale,
                "numeric_weights": numeric_weights,
                "numeric_mean": numeric_mean,
                "numeric_scale": numeric_scale,
                "numeric_score_scale": numeric_score_scale,
            },
        )
    )

    # V6 calibrates the same compact numeric evidence with logistic loss. The
    # fixed blend was required to match or improve V5 on validation and on all
    # three predeclared training-fold assignments before it was frozen.
    numeric_logistic_weights = fit_logistic(
        train_numeric_x, train_labels, V6_NUMERIC_LOGISTIC_L2
    )
    train_numeric_logistic_scores = train_numeric_x @ numeric_logistic_weights
    numeric_logistic_score_scale = float(train_numeric_logistic_scores.std())
    v6_scores = (
        (validation_base_scores / base_score_scale).reshape(len(validation_bank), 4)
        + V5_NONLINEAR_GAMMA * nonlinear_validation_scores
        + V5_ARITHMETIC_GAMMA
        * (validation_arithmetic_x @ arithmetic_weights / arithmetic_score_scale).reshape(
            len(validation_bank), 4
        )
        + V5_ENHANCED_ARITHMETIC_GAMMA
        * (
            validation_enhanced_arithmetic_x
            @ enhanced_arithmetic_weights
            / enhanced_arithmetic_score_scale
        ).reshape(len(validation_bank), 4)
        + V6_NUMERIC_RIDGE_GAMMA
        * (validation_numeric_x @ numeric_weights / numeric_score_scale).reshape(
            len(validation_bank), 4
        )
        + V6_NUMERIC_LOGISTIC_GAMMA
        * (
            validation_numeric_x
            @ numeric_logistic_weights
            / numeric_logistic_score_scale
        ).reshape(len(validation_bank), 4)
    )
    v6_choices = choices_from_answer_groups(
        validation_bank, v6_scores, reducer="max", count_bonus=V6_COUNT_BONUS
    )
    v6_metrics = evaluate("validation", validation_bank, v6_choices, args.data_dir)
    v6_record = {
        "method": "structured_logistic_numeric_fusion",
        "surface_gamma": V4_SURFACE_GAMMA,
        "nonlinear_gamma": V5_NONLINEAR_GAMMA,
        "arithmetic_gamma": V5_ARITHMETIC_GAMMA,
        "enhanced_arithmetic_gamma": V5_ENHANCED_ARITHMETIC_GAMMA,
        "numeric_ridge_gamma": V6_NUMERIC_RIDGE_GAMMA,
        "numeric_logistic_l2": V6_NUMERIC_LOGISTIC_L2,
        "numeric_logistic_gamma": V6_NUMERIC_LOGISTIC_GAMMA,
        "count_bonus": V6_COUNT_BONUS,
        **v6_metrics,
    }
    results.append(v6_record)
    candidates.append(
        (
            v6_record,
            {
                "model_kind": "structured_logistic_fusion",
                "embedding_weights": embedding_weights,
                "embedding_mean": embedding_mean,
                "embedding_scale": embedding_scale,
                "surface_weights": surface_weights,
                "surface_mean": surface_mean,
                "surface_scale": surface_scale,
                "base_score_scale": base_score_scale,
                "random_projection": random_projection,
                "random_weights": random_weights,
                "random_mean": random_mean,
                "random_scale": random_scale,
                "random_score_scale": random_score_scale,
                "arithmetic_weights": arithmetic_weights,
                "arithmetic_mean": arithmetic_mean,
                "arithmetic_scale": arithmetic_scale,
                "arithmetic_score_scale": arithmetic_score_scale,
                "enhanced_arithmetic_weights": enhanced_arithmetic_weights,
                "enhanced_arithmetic_mean": enhanced_arithmetic_mean,
                "enhanced_arithmetic_scale": enhanced_arithmetic_scale,
                "enhanced_arithmetic_score_scale": enhanced_arithmetic_score_scale,
                "numeric_weights": numeric_weights,
                "numeric_logistic_weights": numeric_logistic_weights,
                "numeric_mean": numeric_mean,
                "numeric_scale": numeric_scale,
                "numeric_score_scale": numeric_score_scale,
                "numeric_logistic_score_scale": numeric_logistic_score_scale,
            },
        )
    )

    # The listwise variants start from the ridge solution selected without any
    # validation information (alpha=1.0), then optimize only training labels.
    for feature_set in ("embedding", "combined"):
        train_x, validation_x, mean, scale = prepared[feature_set]
        initial_weights = fit_ridge(train_x, train_labels, alpha=1.0)
        for l2 in LISTWISE_L2:
            weights, training_loss = fit_group_softmax(
                train_x, train_labels, initial_weights, l2=l2
            )
            scores = (validation_x @ weights).reshape(len(validation_bank), 4)
            for temperature in GROUP_TEMPERATURES:
                choices = choices_from_answer_groups(
                    validation_bank, scores, temperature=temperature
                )
                metrics = evaluate("validation", validation_bank, choices, args.data_dir)
                record = {
                    "method": f"{feature_set}_listwise",
                    "l2": l2,
                    "steps": LISTWISE_STEPS,
                    "temperature": temperature,
                    "training_loss": training_loss,
                    **metrics,
                }
                results.append(record)
                candidates.append(
                    (
                        record,
                        {
                            "feature_set": feature_set,
                            "weights": weights,
                            "mean": mean,
                            "scale": scale,
                            "selection_mode": "answer_group",
                            "temperature": temperature,
                        },
                    )
                )

    best, bundle = max(
        candidates,
        key=lambda item: (
            item[0]["accuracy"],
            item[0]["disagreement_accuracy"],
            item[0]["method"] == "structured_logistic_numeric_fusion",
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "model.npz"
    if bundle.get("model_kind") in ("structured_fusion", "structured_logistic_fusion"):
        structured_payload = {
            "feature_version": np.asarray(FEATURE_VERSION),
            "model_kind": np.asarray(bundle["model_kind"]),
            "surface_gamma": np.asarray(V4_SURFACE_GAMMA, dtype=np.float32),
            "nonlinear_gamma": np.asarray(V5_NONLINEAR_GAMMA, dtype=np.float32),
            "arithmetic_gamma": np.asarray(V5_ARITHMETIC_GAMMA, dtype=np.float32),
            "enhanced_arithmetic_gamma": np.asarray(V5_ENHANCED_ARITHMETIC_GAMMA, dtype=np.float32),
            "numeric_gamma": np.asarray(
                V6_NUMERIC_RIDGE_GAMMA if bundle["model_kind"] == "structured_logistic_fusion"
                else V5_NUMERIC_GAMMA, dtype=np.float32
            ),
            "count_bonus": np.asarray(
                V6_COUNT_BONUS if bundle["model_kind"] == "structured_logistic_fusion"
                else V5_COUNT_BONUS, dtype=np.float32
            ),
            "random_feature_scale": np.asarray(RANDOM_FEATURE_SCALE, dtype=np.float32),
            "arithmetic_columns": np.asarray(ARITHMETIC_COLUMNS, dtype=np.int32),
            "embedding_weights": bundle["embedding_weights"],
            "embedding_mean": bundle["embedding_mean"],
            "embedding_scale": bundle["embedding_scale"],
            "surface_weights": bundle["surface_weights"],
            "surface_mean": bundle["surface_mean"],
            "surface_scale": bundle["surface_scale"],
            "base_score_scale": np.asarray(bundle["base_score_scale"], dtype=np.float32),
            "random_projection": bundle["random_projection"],
            "random_weights": bundle["random_weights"],
            "random_mean": bundle["random_mean"],
            "random_scale": bundle["random_scale"],
            "random_score_scale": np.asarray(bundle["random_score_scale"], dtype=np.float32),
            "arithmetic_weights": bundle["arithmetic_weights"],
            "arithmetic_mean": bundle["arithmetic_mean"],
            "arithmetic_scale": bundle["arithmetic_scale"],
            "arithmetic_score_scale": np.asarray(bundle["arithmetic_score_scale"], dtype=np.float32),
            "enhanced_arithmetic_weights": bundle["enhanced_arithmetic_weights"],
            "enhanced_arithmetic_mean": bundle["enhanced_arithmetic_mean"],
            "enhanced_arithmetic_scale": bundle["enhanced_arithmetic_scale"],
            "enhanced_arithmetic_score_scale": np.asarray(
                bundle["enhanced_arithmetic_score_scale"], dtype=np.float32
            ),
            "numeric_weights": bundle["numeric_weights"],
            "numeric_mean": bundle["numeric_mean"],
            "numeric_scale": bundle["numeric_scale"],
            "numeric_score_scale": np.asarray(bundle["numeric_score_scale"], dtype=np.float32),
        }
        if bundle["model_kind"] == "structured_logistic_fusion":
            structured_payload.update(
                numeric_logistic_gamma=np.asarray(V6_NUMERIC_LOGISTIC_GAMMA, dtype=np.float32),
                numeric_logistic_l2=np.asarray(V6_NUMERIC_LOGISTIC_L2, dtype=np.float32),
                numeric_logistic_weights=bundle["numeric_logistic_weights"],
                numeric_logistic_score_scale=np.asarray(
                    bundle["numeric_logistic_score_scale"], dtype=np.float32
                ),
            )
        np.savez(checkpoint_path, **structured_payload)
    elif bundle.get("model_kind") == "nonlinear_arithmetic_fusion":
        np.savez(
            checkpoint_path,
            feature_version=np.asarray(FEATURE_VERSION),
            model_kind=np.asarray("nonlinear_arithmetic_fusion"),
            surface_gamma=np.asarray(V4_SURFACE_GAMMA, dtype=np.float32),
            nonlinear_gamma=np.asarray(V4_NONLINEAR_GAMMA, dtype=np.float32),
            arithmetic_gamma=np.asarray(V4_ARITHMETIC_GAMMA, dtype=np.float32),
            count_bonus=np.asarray(V4_COUNT_BONUS, dtype=np.float32),
            random_feature_scale=np.asarray(RANDOM_FEATURE_SCALE, dtype=np.float32),
            arithmetic_columns=np.asarray(ARITHMETIC_COLUMNS, dtype=np.int32),
            embedding_weights=bundle["embedding_weights"],
            embedding_mean=bundle["embedding_mean"],
            embedding_scale=bundle["embedding_scale"],
            surface_weights=bundle["surface_weights"],
            surface_mean=bundle["surface_mean"],
            surface_scale=bundle["surface_scale"],
            base_score_scale=np.asarray(bundle["base_score_scale"], dtype=np.float32),
            random_projection=bundle["random_projection"],
            random_weights=bundle["random_weights"],
            random_mean=bundle["random_mean"],
            random_scale=bundle["random_scale"],
            random_score_scale=np.asarray(bundle["random_score_scale"], dtype=np.float32),
            arithmetic_weights=bundle["arithmetic_weights"],
            arithmetic_mean=bundle["arithmetic_mean"],
            arithmetic_scale=bundle["arithmetic_scale"],
            arithmetic_score_scale=np.asarray(
                bundle["arithmetic_score_scale"], dtype=np.float32
            ),
        )
    elif bundle.get("model_kind") == "late_fusion":
        np.savez(
            checkpoint_path,
            feature_version=np.asarray(FEATURE_VERSION),
            model_kind=np.asarray("late_fusion"),
            fusion_gamma=np.asarray(bundle["fusion_gamma"], dtype=np.float32),
            count_bonus=np.asarray(bundle["count_bonus"], dtype=np.float32),
            embedding_weights=bundle["embedding_weights"],
            embedding_mean=bundle["embedding_mean"],
            embedding_scale=bundle["embedding_scale"],
            surface_weights=bundle["surface_weights"],
            surface_mean=bundle["surface_mean"],
            surface_scale=bundle["surface_scale"],
        )
    else:
        np.savez(
            checkpoint_path,
            feature_version=np.asarray(FEATURE_VERSION),
            model_kind=np.asarray("single"),
            feature_set=np.asarray(bundle["feature_set"]),
            selection_mode=np.asarray(bundle["selection_mode"]),
            temperature=np.asarray(bundle["temperature"], dtype=np.float32),
            weights=bundle["weights"],
            mean=bundle["mean"],
            scale=bundle["scale"],
        )
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    summary = {
        "selection_rule": (
            "highest full-validation accuracy; disagreement accuracy breaks ties; "
            "the preregistered V6 repeated-CV winner breaks an exact remaining tie"
        ),
        "training_split": "train",
        "selection_split": "validation",
        "test_used": False,
        "seed": None,
        "deterministic": True,
        "alpha_grid": list(ALPHAS),
        "group_temperature_grid": list(GROUP_TEMPERATURES),
        "fusion_gamma_grid": list(FUSION_GAMMAS),
        "count_bonus_grid": list(COUNT_BONUSES),
        "fusion_embedding_alpha": FUSION_EMBEDDING_ALPHA,
        "fusion_surface_alpha": FUSION_SURFACE_ALPHA,
        "random_feature_seed": RANDOM_FEATURE_SEED,
        "random_feature_dim": RANDOM_FEATURE_DIM,
        "random_feature_scale": RANDOM_FEATURE_SCALE,
        "random_feature_alpha": RANDOM_FEATURE_ALPHA,
        "arithmetic_columns": list(ARITHMETIC_COLUMNS),
        "arithmetic_alpha": ARITHMETIC_ALPHA,
        "v4_fusion": {
            "surface_gamma": V4_SURFACE_GAMMA,
            "nonlinear_gamma": V4_NONLINEAR_GAMMA,
            "arithmetic_gamma": V4_ARITHMETIC_GAMMA,
            "count_bonus": V4_COUNT_BONUS,
        },
        "v5_fusion": {
            "enhanced_arithmetic_alpha": ENHANCED_ARITHMETIC_ALPHA,
            "numeric_alpha": NUMERIC_ALPHA,
            "surface_gamma": V4_SURFACE_GAMMA,
            "nonlinear_gamma": V5_NONLINEAR_GAMMA,
            "arithmetic_gamma": V5_ARITHMETIC_GAMMA,
            "enhanced_arithmetic_gamma": V5_ENHANCED_ARITHMETIC_GAMMA,
            "numeric_gamma": V5_NUMERIC_GAMMA,
            "count_bonus": V5_COUNT_BONUS,
        },
        "v6_fusion": {
            "numeric_ridge_gamma": V6_NUMERIC_RIDGE_GAMMA,
            "numeric_logistic_l2": V6_NUMERIC_LOGISTIC_L2,
            "numeric_logistic_gamma": V6_NUMERIC_LOGISTIC_GAMMA,
            "count_bonus": V6_COUNT_BONUS,
        },
        "listwise_l2_grid": list(LISTWISE_L2),
        "listwise_steps": LISTWISE_STEPS,
        "listwise_learning_rate": LISTWISE_LEARNING_RATE,
        "feature_sets": list(FEATURE_SETS),
        "selected": best,
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": digest,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "training_seconds": time.perf_counter() - started,
        "validation_results": results,
    }
    (args.output_dir / "training-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
