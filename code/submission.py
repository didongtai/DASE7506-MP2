"""Frozen lightweight verifier for DASE7506 MP2."""

from pathlib import Path
import re

import numpy as np

from solution_utils import (
    FEATURE_VERSION,
    build_feature_blocks,
    choices_from_answer_groups,
    choices_from_scores,
    majority_choice,
    normalize_features,
    parsed_answers,
    program_search_features,
    read_once_expression_features,
    select_matrix,
    trace_execution_features,
)


def _group_structure(bank):
    equal = np.zeros((len(bank), 4, 4), dtype=bool)
    representative = np.ones((len(bank), 4), dtype=bool)
    counts = np.zeros((len(bank), 4), dtype=np.float32)
    for row_index, row in enumerate(bank):
        answers = parsed_answers(row)
        for candidate, answer in enumerate(answers):
            equal[row_index, candidate] = [answer is not None and answer == other for other in answers]
            representative[row_index, candidate] = answer is not None and not any(
                answers[earlier] == answer for earlier in range(candidate)
            )
            counts[row_index, candidate] = equal[row_index, candidate].sum()
    return equal, representative, counts


def _kernel_group_features(
    bank, blocks, raw_embeddings, components, scores, count_bonus, projection,
    projected_embeddings=None,
):
    equal, representative, counts = _group_structure(bank)
    component = np.stack(components, axis=2)
    surface = blocks["surface"][:, :, :19]
    arithmetic = blocks["arithmetic_v2"]
    numeric = blocks["numeric"]
    width = 2 * (component.shape[2] + 19 + 12) + 31 + 2
    base = np.zeros((len(bank), 4, width), dtype=np.float32)
    grouped = np.full((len(bank), 4), -np.inf, dtype=np.float32)
    for i in range(len(bank)):
        for j in np.flatnonzero(representative[i]):
            members = equal[i, j]
            base[i, j] = np.concatenate((
                component[i, members].max(axis=0), component[i, members].mean(axis=0),
                surface[i, members].max(axis=0), surface[i, members].mean(axis=0),
                arithmetic[i, members].max(axis=0), arithmetic[i, members].mean(axis=0),
                numeric[i, j], np.asarray([members.sum() / 4.0, np.log(members.sum())]),
            ))
            grouped[i, j] = scores[i, members].max() + count_bonus * np.log(members.sum())

    extras = np.zeros((len(bank), 4, 19), dtype=np.float32)
    for i in range(len(bank)):
        valid_scores = grouped[i, representative[i]]
        best, mean = valid_scores.max(), valid_scores.mean()
        ordered = np.sort(np.where(representative[i], grouped[i], -1e9))
        margin = ordered[-1] - ordered[-2]
        n_groups = int(representative[i].sum())
        component_max = component[i, representative[i]].max(axis=0)
        for j in np.flatnonzero(representative[i]):
            extras[i, j, 0] = grouped[i, j]
            extras[i, j, 1] = grouped[i, j] - best
            extras[i, j, 2] = grouped[i, j] - mean
            extras[i, j, 3] = margin
            extras[i, j, 4] = n_groups / 4.0
            extras[i, j, 5] = counts[i, j] / 4.0
            extras[i, j, 6] = float(counts[i, j] == counts[i].max())
            extras[i, j, 7] = float(counts[i, j] > 1)
            extras[i, j, 8:14] = component[i, j] - component_max
            extras[i, j, 14 + min(n_groups, 4) - 1] = 1.0
            extras[i, j, 18] = float(grouped[i, j] == best)

    if projected_embeddings is None:
        projected = np.tanh(.5 * (np.asarray(raw_embeddings).reshape(-1, 1024) @ projection))
        projected = projected.reshape(len(bank), 4, -1)
    else:
        projected = np.asarray(projected_embeddings, dtype=np.float32)
    dimension = projected.shape[2]
    semantic = np.zeros((len(bank), 4, 2 * dimension), dtype=np.float32)
    for i in range(len(bank)):
        for j in np.flatnonzero(representative[i]):
            members = equal[i, j]
            semantic[i, j, :dimension] = projected[i, members].mean(axis=0)
            semantic[i, j, dimension:] = projected[i, members].max(axis=0)
    return np.concatenate((base, extras, semantic), axis=2), representative, grouped


def _checkpoint_file(checkpoint):
    path = Path(checkpoint)
    return path / "model.npz" if path.is_dir() else path


def _score_margin(scores, representative):
    ordered = np.sort(np.where(representative, scores, -np.inf), axis=1)
    result = ordered[:, -1] - ordered[:, -2]
    return np.where(np.isfinite(result), result, 0.0)


def _router_topic_features(bank):
    phrase_sets = (
        ("percent", "%", "discount", "raise"),
        ("week", "month", "year", "day", "hour", "minute"),
        ("$", "dollar", "cent", "cost", "price", "profit"),
        ("each", " per ", "every"),
        ("left", "remaining", "remain", "rest"),
        ("twice", "times", "double", "triple", "half"),
        ("total", "altogether", "in all", "combined"),
        ("more", "less", "fewer", "difference"),
        ("dozen", "pair", "pack", "box", "carton"),
        ("speed", "rate", "miles per", "per hour"),
    )
    result = np.zeros((len(bank), len(phrase_sets) + 3), dtype=np.float32)
    for i, row in enumerate(bank):
        question = row["question"].lower()
        for j, phrases in enumerate(phrase_sets):
            result[i, j] = float(any(phrase in question for phrase in phrases))
        result[i, -3] = np.log1p(len(re.findall(r"\d+(?:\.\d+)?", question)))
        result[i, -2] = np.log1p(len(question.split()))
        result[i, -1] = float("?" in question)
    return result


def _disagreement_router_features(bank, embeddings, representative, counts,
                                  base, current, alternative, context):
    current_choice = current.argmax(axis=1)
    alternative_choice = alternative.argmax(axis=1)
    row_numbers = np.arange(len(bank))
    result = np.zeros((len(bank), 184), dtype=np.float32)
    result[:, 0] = _score_margin(base, representative)
    result[:, 1] = _score_margin(current, representative)
    result[:, 2] = _score_margin(alternative, representative)
    result[:, 3] = result[:, 2] - result[:, 1]
    result[:, 4] = representative.sum(axis=1) / 4.0
    result[:, 5] = counts[row_numbers, current_choice] / 4.0
    result[:, 6] = counts[row_numbers, alternative_choice] / 4.0
    result[:, 7] = counts[row_numbers, current_choice] == counts.max(axis=1)
    result[:, 8] = counts[row_numbers, alternative_choice] == counts.max(axis=1)
    result[:, 159:172] = _router_topic_features(bank)
    result[:, 172:184] = context
    return result


def _router_raw_predict(features, nodes, node_counts, baseline):
    result = np.full(len(features), baseline, dtype=np.float64)
    offset = 0
    for count in node_counts:
        tree = nodes[offset:offset + int(count)]
        offset += int(count)
        for row_index, row in enumerate(features):
            node_index = 0
            while not tree[node_index]["is_leaf"]:
                node = tree[node_index]
                value = row[int(node["feature_idx"])]
                if np.isnan(value):
                    go_left = bool(node["missing_go_to_left"])
                else:
                    go_left = value <= float(node["num_threshold"])
                node_index = int(node["left"] if go_left else node["right"])
            result[row_index] += float(tree[node_index]["value"])
    return result


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order."""
    if checkpoint is None:
        return [majority_choice(row) for row in bank]

    with np.load(_checkpoint_file(checkpoint), allow_pickle=False) as model:
        version = int(model["feature_version"])
        if version != FEATURE_VERSION:
            raise ValueError(f"Checkpoint feature version {version} is unsupported")
        model_kind = str(model["model_kind"]) if "model_kind" in model.files else "single"
        if model_kind in ("structured_fusion", "structured_logistic_fusion", "kernel_router_v8", "kernel_router_v9", "dual_objective_v10", "expression_ensemble_v12", "disagreement_router_v17", "multi_expert_router_v20", "listwise_trace_v23", "listwise_route_v24", "listwise_consensus_v25", "expert_consensus_v27", "listwise_capacity_v28", "program_search_v32"):
            surface_gamma = float(model["surface_gamma"])
            nonlinear_gamma = float(model["nonlinear_gamma"])
            arithmetic_gamma = float(model["arithmetic_gamma"])
            enhanced_arithmetic_gamma = float(model["enhanced_arithmetic_gamma"])
            numeric_gamma = float(model["numeric_gamma"])
            count_bonus = float(model["count_bonus"])
            random_feature_scale = float(model["random_feature_scale"])
            arithmetic_columns = model["arithmetic_columns"].astype(np.int32, copy=False)
            embedding_weights = model["embedding_weights"].astype(np.float32, copy=False)
            embedding_mean = model["embedding_mean"].astype(np.float32, copy=False)
            embedding_scale = model["embedding_scale"].astype(np.float32, copy=False)
            surface_weights = model["surface_weights"].astype(np.float32, copy=False)
            surface_mean = model["surface_mean"].astype(np.float32, copy=False)
            surface_scale = model["surface_scale"].astype(np.float32, copy=False)
            base_score_scale = float(model["base_score_scale"])
            random_projection = model["random_projection"].astype(np.float32, copy=False)
            random_weights = model["random_weights"].astype(np.float32, copy=False)
            random_mean = model["random_mean"].astype(np.float32, copy=False)
            random_scale = model["random_scale"].astype(np.float32, copy=False)
            random_score_scale = float(model["random_score_scale"])
            arithmetic_weights = model["arithmetic_weights"].astype(np.float32, copy=False)
            arithmetic_mean = model["arithmetic_mean"].astype(np.float32, copy=False)
            arithmetic_scale = model["arithmetic_scale"].astype(np.float32, copy=False)
            arithmetic_score_scale = float(model["arithmetic_score_scale"])
            enhanced_arithmetic_weights = model["enhanced_arithmetic_weights"].astype(np.float32, copy=False)
            enhanced_arithmetic_mean = model["enhanced_arithmetic_mean"].astype(np.float32, copy=False)
            enhanced_arithmetic_scale = model["enhanced_arithmetic_scale"].astype(np.float32, copy=False)
            enhanced_arithmetic_score_scale = float(model["enhanced_arithmetic_score_scale"])
            numeric_weights = model["numeric_weights"].astype(np.float32, copy=False)
            numeric_mean = model["numeric_mean"].astype(np.float32, copy=False)
            numeric_scale = model["numeric_scale"].astype(np.float32, copy=False)
            numeric_score_scale = float(model["numeric_score_scale"])
            if model_kind in ("structured_logistic_fusion", "kernel_router_v8", "kernel_router_v9", "dual_objective_v10", "expression_ensemble_v12", "disagreement_router_v17", "multi_expert_router_v20", "listwise_trace_v23", "listwise_route_v24", "listwise_consensus_v25", "expert_consensus_v27", "listwise_capacity_v28", "program_search_v32"):
                numeric_logistic_gamma = float(model["numeric_logistic_gamma"])
                numeric_logistic_weights = model["numeric_logistic_weights"].astype(
                    np.float32, copy=False
                )
                numeric_logistic_score_scale = float(
                    model["numeric_logistic_score_scale"]
                )
            if model_kind in ("kernel_router_v8", "kernel_router_v9", "dual_objective_v10", "expression_ensemble_v12", "disagreement_router_v17", "multi_expert_router_v20", "listwise_trace_v23", "listwise_route_v24", "listwise_consensus_v25", "expert_consensus_v27", "listwise_capacity_v28", "program_search_v32"):
                kernel_feature_mean = model["kernel_feature_mean"].astype(np.float32, copy=False)
                kernel_feature_scale = model["kernel_feature_scale"].astype(np.float32, copy=False)
                kernel_support_vectors = model["kernel_support_vectors"].astype(np.float32, copy=False)
                kernel_dual_coef = model["kernel_dual_coef"].astype(np.float32, copy=False)
                kernel_intercept = float(model["kernel_intercept"])
                kernel_gamma = float(model["kernel_gamma"])
                kernel_blend = float(model["kernel_blend"])
                kernel_support_norm = model["kernel_support_norm"].astype(np.float32, copy=False)
            if model_kind in ("kernel_router_v9", "dual_objective_v10", "expression_ensemble_v12", "disagreement_router_v17", "multi_expert_router_v20", "listwise_trace_v23", "listwise_route_v24", "listwise_consensus_v25", "expert_consensus_v27", "listwise_capacity_v28", "program_search_v32"):
                semantic_pca_mean = model["semantic_pca_mean"].astype(np.float32, copy=False)
                semantic_pca_components = model["semantic_pca_components"].astype(np.float32, copy=False)
            if model_kind == "expression_ensemble_v12":
                expression_fraction = float(model["expression_fraction"])
                expression_kernel_feature_mean = model["expression_kernel_feature_mean"].astype(np.float32, copy=False)
                expression_kernel_feature_scale = model["expression_kernel_feature_scale"].astype(np.float32, copy=False)
                expression_kernel_support_vectors = model["expression_kernel_support_vectors"].astype(np.float32, copy=False)
                expression_kernel_dual_coef = model["expression_kernel_dual_coef"].astype(np.float32, copy=False)
                expression_kernel_intercept = float(model["expression_kernel_intercept"])
                expression_kernel_gamma = float(model["expression_kernel_gamma"])
                expression_kernel_support_norm = model["expression_kernel_support_norm"].astype(np.float32, copy=False)
            if model_kind in ("disagreement_router_v17", "multi_expert_router_v20", "expert_consensus_v27"):
                expression_kernel_blend = float(model["expression_kernel_blend"])
                expression_kernel_feature_mean = model["expression_kernel_feature_mean"].astype(np.float32, copy=False)
                expression_kernel_feature_scale = model["expression_kernel_feature_scale"].astype(np.float32, copy=False)
                expression_kernel_support_vectors = model["expression_kernel_support_vectors"].astype(np.float32, copy=False)
                expression_kernel_dual_coef = model["expression_kernel_dual_coef"].astype(np.float32, copy=False)
                expression_kernel_intercept = float(model["expression_kernel_intercept"])
                expression_kernel_gamma = float(model["expression_kernel_gamma"])
                expression_kernel_support_norm = model["expression_kernel_support_norm"].astype(np.float32, copy=False)
                context_pca_mean = model["context_pca_mean"].astype(np.float32, copy=False)
                context_pca_components = model["context_pca_components"].astype(np.float32, copy=False)
            if model_kind == "disagreement_router_v17":
                router_baseline = float(model["router_baseline"])
                router_threshold = float(model["router_threshold"])
                router_node_counts = model["router_node_counts"].astype(np.int32, copy=False)
                router_nodes = model["router_nodes"]
            if model_kind in ("multi_expert_router_v20", "expert_consensus_v27"):
                wide_kernel_blend = float(model["wide_kernel_blend"])
                wide_kernel_feature_mean = model["wide_kernel_feature_mean"].astype(np.float32, copy=False)
                wide_kernel_feature_scale = model["wide_kernel_feature_scale"].astype(np.float32, copy=False)
                wide_kernel_support_vectors = model["wide_kernel_support_vectors"].astype(np.float32, copy=False)
                wide_kernel_dual_coef = model["wide_kernel_dual_coef"].astype(np.float32, copy=False)
                wide_kernel_intercept = float(model["wide_kernel_intercept"])
                wide_kernel_gamma = float(model["wide_kernel_gamma"])
                wide_kernel_support_norm = model["wide_kernel_support_norm"].astype(np.float32, copy=False)
                expression_router_baseline = float(model["expression_router_baseline"])
                expression_router_threshold = float(model["expression_router_threshold"])
                expression_router_node_counts = model["expression_router_node_counts"].astype(np.int32, copy=False)
                expression_router_nodes = model["expression_router_nodes"]
                wide_router_feature_mean = model["wide_router_feature_mean"].astype(np.float32, copy=False)
                wide_router_feature_scale = model["wide_router_feature_scale"].astype(np.float32, copy=False)
                wide_router_weights = model["wide_router_weights"].astype(np.float32, copy=False)
                wide_router_intercept = float(model["wide_router_intercept"])
                wide_router_threshold = float(model["wide_router_threshold"])
            if model_kind == "listwise_trace_v23":
                listwise_blend = float(model["listwise_blend"])
                listwise_feature_mean = model["listwise_feature_mean"].astype(np.float32, copy=False)
                listwise_feature_scale = model["listwise_feature_scale"].astype(np.float32, copy=False)
                listwise_first_weights = model["listwise_first_weights"].astype(np.float32, copy=False)
                listwise_first_bias = model["listwise_first_bias"].astype(np.float32, copy=False)
                listwise_second_weights = model["listwise_second_weights"].astype(np.float32, copy=False)
                listwise_second_bias = model["listwise_second_bias"].astype(np.float32, copy=False)
            if model_kind == "listwise_route_v24":
                listwise_feature_mean = model["listwise_feature_mean"].astype(np.float32, copy=False)
                listwise_feature_scale = model["listwise_feature_scale"].astype(np.float32, copy=False)
                listwise_weights = model["listwise_weights"].astype(np.float32, copy=False)
                listwise_bias = float(model["listwise_bias"].reshape(-1)[0])
                listwise_advantage_threshold = float(model["listwise_advantage_threshold"])
                listwise_margin_threshold = float(model["listwise_margin_threshold"])
            if model_kind == "listwise_consensus_v25":
                listwise_models = []
                for name in ("all", "no_trace", "no_semantic", "semantic_trace"):
                    prefix = f"listwise_{name}"
                    listwise_models.append((
                        model[f"{prefix}_feature_mean"].astype(np.float32, copy=False),
                        model[f"{prefix}_feature_scale"].astype(np.float32, copy=False),
                        model[f"{prefix}_weights"].astype(np.float32, copy=False),
                        float(model[f"{prefix}_bias"].reshape(-1)[0]),
                    ))
                listwise_minimum_votes = int(model["listwise_minimum_votes"])
                listwise_confidence_threshold = float(
                    model["listwise_confidence_threshold"]
                )
                listwise_margin_threshold = float(model["listwise_margin_threshold"])
            if model_kind == "expert_consensus_v27":
                v23_blend = float(model["v23_listwise_blend"])
                v23_mean = model["v23_listwise_feature_mean"].astype(np.float32, copy=False)
                v23_scale = model["v23_listwise_feature_scale"].astype(np.float32, copy=False)
                v23_first_weights = model["v23_listwise_first_weights"].astype(np.float32, copy=False)
                v23_first_bias = model["v23_listwise_first_bias"].astype(np.float32, copy=False)
                v23_second_weights = model["v23_listwise_second_weights"].astype(np.float32, copy=False)
                v23_second_bias = model["v23_listwise_second_bias"].astype(np.float32, copy=False)
                v24_mean = model["v24_listwise_feature_mean"].astype(np.float32, copy=False)
                v24_scale = model["v24_listwise_feature_scale"].astype(np.float32, copy=False)
                v24_weights = model["v24_listwise_weights"].astype(np.float32, copy=False)
                v24_bias = float(model["v24_listwise_bias"].reshape(-1)[0])
                v24_advantage_threshold = float(model["v24_listwise_advantage_threshold"])
                v24_margin_threshold = float(model["v24_listwise_margin_threshold"])
                v25_models = []
                for name in ("all", "no_trace", "no_semantic", "semantic_trace"):
                    prefix = f"v25_listwise_{name}"
                    v25_models.append((
                        model[f"{prefix}_feature_mean"].astype(np.float32, copy=False),
                        model[f"{prefix}_feature_scale"].astype(np.float32, copy=False),
                        model[f"{prefix}_weights"].astype(np.float32, copy=False),
                        float(model[f"{prefix}_bias"].reshape(-1)[0]),
                    ))
                v25_minimum_votes = int(model["v25_listwise_minimum_votes"])
                v25_confidence_threshold = float(
                    model["v25_listwise_confidence_threshold"]
                )
                v25_margin_threshold = float(model["v25_listwise_margin_threshold"])
                v27_minimum_votes = int(model["v27_minimum_votes"])
            if model_kind == "listwise_capacity_v28":
                listwise_dimension = int(model["listwise_dimension"])
                listwise_blend = float(model["listwise_blend"])
                listwise_feature_mean = model["listwise_feature_mean"].astype(np.float32, copy=False)
                listwise_feature_scale = model["listwise_feature_scale"].astype(np.float32, copy=False)
                listwise_weights = model["listwise_weights"].astype(np.float32, copy=False)
                listwise_bias = float(model["listwise_bias"].reshape(-1)[0])
            if model_kind == "program_search_v32":
                listwise_dimension = int(model["listwise_dimension"])
                listwise_blend = float(model["listwise_blend"])
                listwise_feature_mean = model["listwise_feature_mean"].astype(
                    np.float32, copy=False
                )
                listwise_feature_scale = model["listwise_feature_scale"].astype(
                    np.float32, copy=False
                )
                listwise_first_weights = model["listwise_first_weights"].astype(
                    np.float32, copy=False
                )
                listwise_first_bias = model["listwise_first_bias"].astype(
                    np.float32, copy=False
                )
                listwise_second_weights = model["listwise_second_weights"].astype(
                    np.float32, copy=False
                )
                listwise_second_bias = model["listwise_second_bias"].astype(
                    np.float32, copy=False
                )
            if model_kind == "dual_objective_v10":
                pair_feature_mean = model["pair_feature_mean"].astype(np.float32, copy=False)
                pair_feature_scale = model["pair_feature_scale"].astype(np.float32, copy=False)
                pair_support_vectors = model["pair_support_vectors"].astype(np.float32, copy=False)
                pair_dual_coef = model["pair_dual_coef"].astype(np.float32, copy=False)
                pair_intercept = float(model["pair_intercept"])
                pair_gamma = float(model["pair_gamma"])
                pair_blend = float(model["pair_blend"])
                pair_support_norm = model["pair_support_norm"].astype(np.float32, copy=False)
        elif model_kind == "nonlinear_arithmetic_fusion":
            surface_gamma = float(model["surface_gamma"])
            nonlinear_gamma = float(model["nonlinear_gamma"])
            arithmetic_gamma = float(model["arithmetic_gamma"])
            count_bonus = float(model["count_bonus"])
            random_feature_scale = float(model["random_feature_scale"])
            arithmetic_columns = model["arithmetic_columns"].astype(np.int32, copy=False)
            embedding_weights = model["embedding_weights"].astype(np.float32, copy=False)
            embedding_mean = model["embedding_mean"].astype(np.float32, copy=False)
            embedding_scale = model["embedding_scale"].astype(np.float32, copy=False)
            surface_weights = model["surface_weights"].astype(np.float32, copy=False)
            surface_mean = model["surface_mean"].astype(np.float32, copy=False)
            surface_scale = model["surface_scale"].astype(np.float32, copy=False)
            base_score_scale = float(model["base_score_scale"])
            random_projection = model["random_projection"].astype(np.float32, copy=False)
            random_weights = model["random_weights"].astype(np.float32, copy=False)
            random_mean = model["random_mean"].astype(np.float32, copy=False)
            random_scale = model["random_scale"].astype(np.float32, copy=False)
            random_score_scale = float(model["random_score_scale"])
            arithmetic_weights = model["arithmetic_weights"].astype(np.float32, copy=False)
            arithmetic_mean = model["arithmetic_mean"].astype(np.float32, copy=False)
            arithmetic_scale = model["arithmetic_scale"].astype(np.float32, copy=False)
            arithmetic_score_scale = float(model["arithmetic_score_scale"])
        elif model_kind == "late_fusion":
            fusion_gamma = float(model["fusion_gamma"])
            count_bonus = float(model["count_bonus"])
            embedding_weights = model["embedding_weights"].astype(np.float32, copy=False)
            embedding_mean = model["embedding_mean"].astype(np.float32, copy=False)
            embedding_scale = model["embedding_scale"].astype(np.float32, copy=False)
            surface_weights = model["surface_weights"].astype(np.float32, copy=False)
            surface_mean = model["surface_mean"].astype(np.float32, copy=False)
            surface_scale = model["surface_scale"].astype(np.float32, copy=False)
        elif model_kind != "single":
            raise ValueError(f"Unknown checkpoint model kind: {model_kind}")
        else:
            feature_set = str(model["feature_set"])
            selection_mode = str(model["selection_mode"]) if "selection_mode" in model.files else "candidate"
            temperature = float(model["temperature"]) if "temperature" in model.files else 1.0
            weights = model["weights"].astype(np.float32, copy=False)
            mean = model["mean"].astype(np.float32, copy=False)
            scale = model["scale"].astype(np.float32, copy=False)

    blocks = build_feature_blocks(bank, arrays["embeddings"])
    if model_kind in ("structured_fusion", "structured_logistic_fusion", "kernel_router_v8", "kernel_router_v9", "dual_objective_v10", "expression_ensemble_v12", "disagreement_router_v17", "multi_expert_router_v20", "listwise_trace_v23", "listwise_route_v24", "listwise_consensus_v25", "expert_consensus_v27", "listwise_capacity_v28", "program_search_v32"):
        embedding_x = select_matrix(blocks, "embedding")
        embedding_x, _, _ = normalize_features(
            embedding_x, embedding_mean, embedding_scale
        )
        surface_x = select_matrix(blocks, "surface")
        surface_x, _, _ = normalize_features(surface_x, surface_mean, surface_scale)
        base_scores = (
            embedding_x @ embedding_weights
            + surface_gamma * (surface_x @ surface_weights)
        ) / base_score_scale
        random_x = np.tanh(
            random_feature_scale
            * (blocks["embedding"].reshape(-1, 1024) @ random_projection)
        )
        random_x, _, _ = normalize_features(random_x, random_mean, random_scale)
        nonlinear_scores = random_x @ random_weights / random_score_scale
        arithmetic_x = select_matrix(blocks, "arithmetic")[:, arithmetic_columns]
        arithmetic_x, _, _ = normalize_features(
            arithmetic_x, arithmetic_mean, arithmetic_scale
        )
        arithmetic_scores = arithmetic_x @ arithmetic_weights / arithmetic_score_scale
        enhanced_arithmetic_x = select_matrix(blocks, "arithmetic_v2")
        enhanced_arithmetic_x, _, _ = normalize_features(
            enhanced_arithmetic_x,
            enhanced_arithmetic_mean,
            enhanced_arithmetic_scale,
        )
        enhanced_arithmetic_scores = (
            enhanced_arithmetic_x @ enhanced_arithmetic_weights
            / enhanced_arithmetic_score_scale
        )
        numeric_x = select_matrix(blocks, "numeric")
        numeric_x, _, _ = normalize_features(numeric_x, numeric_mean, numeric_scale)
        numeric_scores = numeric_x @ numeric_weights / numeric_score_scale
        scores = (
            base_scores
            + nonlinear_gamma * nonlinear_scores
            + arithmetic_gamma * arithmetic_scores
            + enhanced_arithmetic_gamma * enhanced_arithmetic_scores
            + numeric_gamma * numeric_scores
        )
        numeric_logistic_scores = np.zeros_like(numeric_scores)
        if model_kind in ("structured_logistic_fusion", "kernel_router_v8", "kernel_router_v9", "dual_objective_v10", "expression_ensemble_v12", "disagreement_router_v17", "multi_expert_router_v20", "listwise_trace_v23", "listwise_route_v24", "listwise_consensus_v25", "expert_consensus_v27", "listwise_capacity_v28", "program_search_v32"):
            numeric_logistic_scores = (
                numeric_x @ numeric_logistic_weights / numeric_logistic_score_scale
            )
            scores += numeric_logistic_gamma * numeric_logistic_scores
        scores = scores.reshape(len(bank), 4)
        if model_kind in ("kernel_router_v8", "kernel_router_v9", "dual_objective_v10", "expression_ensemble_v12", "disagreement_router_v17", "multi_expert_router_v20", "listwise_trace_v23", "listwise_route_v24", "listwise_consensus_v25", "expert_consensus_v27", "listwise_capacity_v28", "program_search_v32"):
            components = (
                base_scores.reshape(len(bank), 4),
                nonlinear_scores.reshape(len(bank), 4),
                arithmetic_scores.reshape(len(bank), 4),
                enhanced_arithmetic_scores.reshape(len(bank), 4),
                numeric_scores.reshape(len(bank), 4),
                numeric_logistic_scores.reshape(len(bank), 4),
            )
            projected_embeddings = None
            if model_kind in ("kernel_router_v9", "dual_objective_v10", "expression_ensemble_v12", "disagreement_router_v17", "multi_expert_router_v20", "listwise_trace_v23", "listwise_route_v24", "listwise_consensus_v25", "expert_consensus_v27", "listwise_capacity_v28", "program_search_v32"):
                relative = arrays["embeddings"] - arrays["embeddings"].mean(axis=1, keepdims=True)
                relative /= np.maximum(np.linalg.norm(relative, axis=2, keepdims=True), 1e-6)
                projected_embeddings = (
                    (relative.reshape(-1, 1024) - semantic_pca_mean)
                    @ semantic_pca_components.T
                ).reshape(len(bank), 4, -1)
            features, representative, grouped = _kernel_group_features(
                bank, blocks, arrays["embeddings"], components, scores,
                count_bonus, random_projection, projected_embeddings,
            )
            rows = representative.reshape(-1)
            x = features.reshape(-1, features.shape[2])[rows]
            x = (x - kernel_feature_mean) / kernel_feature_scale
            query_norm = np.square(x).sum(axis=1, keepdims=True)
            distance = np.maximum(
                query_norm + kernel_support_norm[None, :] - 2.0 * (x @ kernel_support_vectors.T),
                0.0,
            )
            decision = np.exp(-kernel_gamma * distance) @ kernel_dual_coef + kernel_intercept
            if model_kind in ("expression_ensemble_v12", "disagreement_router_v17", "multi_expert_router_v20", "expert_consensus_v27"):
                expression = read_once_expression_features(bank)
                expression_features = np.concatenate((features, expression), axis=2)
                expression_x = expression_features.reshape(-1, expression_features.shape[2])[rows]
                expression_x = (
                    expression_x - expression_kernel_feature_mean
                ) / expression_kernel_feature_scale
                expression_query_norm = np.square(expression_x).sum(axis=1, keepdims=True)
                expression_distance = np.maximum(
                    expression_query_norm + expression_kernel_support_norm[None, :]
                    - 2.0 * (expression_x @ expression_kernel_support_vectors.T),
                    0.0,
                )
                expression_decision = (
                    np.exp(-expression_kernel_gamma * expression_distance)
                    @ expression_kernel_dual_coef
                    + expression_kernel_intercept
                )
            if model_kind in ("multi_expert_router_v20", "expert_consensus_v27"):
                wide_x = features.reshape(-1, features.shape[2])[rows]
                wide_x = (wide_x - wide_kernel_feature_mean) / wide_kernel_feature_scale
                wide_query_norm = np.square(wide_x).sum(axis=1, keepdims=True)
                wide_distance = np.maximum(
                    wide_query_norm + wide_kernel_support_norm[None, :]
                    - 2.0 * (wide_x @ wide_kernel_support_vectors.T),
                    0.0,
                )
                wide_decision = (
                    np.exp(-wide_kernel_gamma * wide_distance)
                    @ wide_kernel_dual_coef + wide_kernel_intercept
                )
                current = grouped.copy()
                expression_alternative = grouped.copy()
                wide_alternative = grouped.copy()
                current.reshape(-1)[rows] += kernel_blend * decision
                expression_alternative.reshape(-1)[rows] += (
                    expression_kernel_blend * expression_decision
                )
                wide_alternative.reshape(-1)[rows] += wide_kernel_blend * wide_decision
                current_choice = current.argmax(axis=1)
                expression_choice = expression_alternative.argmax(axis=1)
                wide_choice = wide_alternative.argmax(axis=1)
                expression_disagreement = current_choice != expression_choice
                wide_disagreement = current_choice != wide_choice
                if (model_kind == "multi_expert_router_v20"
                        and not (expression_disagreement.any() or wide_disagreement.any())):
                    return current_choice.astype(int).tolist()

                _, _, counts = _group_structure(bank)
                context = (
                    (arrays["embeddings"].mean(axis=1) - context_pca_mean)
                    @ context_pca_components.T
                )
                result = current_choice.copy()
                if wide_disagreement.any():
                    wide_router_x = _disagreement_router_features(
                        bank, arrays["embeddings"], representative, counts,
                        grouped, current, wide_alternative, context,
                    )
                    standardized = (
                        wide_router_x[wide_disagreement] - wide_router_feature_mean
                    ) / wide_router_feature_scale
                    raw = standardized @ wide_router_weights + wide_router_intercept
                    probability = 1.0 / (1.0 + np.exp(-raw))
                    route = np.zeros(len(bank), dtype=bool)
                    route[wide_disagreement] = probability >= wide_router_threshold
                    result = np.where(route, wide_choice, result)
                if expression_disagreement.any():
                    expression_router_x = _disagreement_router_features(
                        bank, arrays["embeddings"], representative, counts,
                        grouped, current, expression_alternative, context,
                    )
                    raw = _router_raw_predict(
                        expression_router_x[expression_disagreement],
                        expression_router_nodes, expression_router_node_counts,
                        expression_router_baseline,
                    )
                    probability = 1.0 / (1.0 + np.exp(-raw))
                    route = np.zeros(len(bank), dtype=bool)
                    route[expression_disagreement] = (
                        probability >= expression_router_threshold
                    )
                    result = np.where(route, expression_choice, result)
                if model_kind == "multi_expert_router_v20":
                    return result.astype(int).tolist()
                choice20 = result
            if model_kind == "disagreement_router_v17":
                current = grouped.copy()
                alternative = grouped.copy()
                current.reshape(-1)[rows] += kernel_blend * decision
                alternative.reshape(-1)[rows] += expression_kernel_blend * expression_decision
                current_choice = current.argmax(axis=1)
                alternative_choice = alternative.argmax(axis=1)
                disagreement = current_choice != alternative_choice
                if not disagreement.any():
                    return current_choice.astype(int).tolist()

                _, _, counts = _group_structure(bank)
                context = (
                    (arrays["embeddings"].mean(axis=1) - context_pca_mean)
                    @ context_pca_components.T
                )
                router_x = np.zeros((len(bank), 184), dtype=np.float32)
                router_x[:, 0] = _score_margin(grouped, representative)
                router_x[:, 1] = _score_margin(current, representative)
                router_x[:, 2] = _score_margin(alternative, representative)
                router_x[:, 3] = router_x[:, 2] - router_x[:, 1]
                router_x[:, 4] = representative.sum(axis=1) / 4.0
                row_numbers = np.arange(len(bank))
                router_x[:, 5] = counts[row_numbers, current_choice] / 4.0
                router_x[:, 6] = counts[row_numbers, alternative_choice] / 4.0
                router_x[:, 7] = (
                    counts[row_numbers, current_choice] == counts.max(axis=1)
                )
                router_x[:, 8] = (
                    counts[row_numbers, alternative_choice] == counts.max(axis=1)
                )
                router_x[:, 159:172] = _router_topic_features(bank)
                router_x[:, 172:184] = context
                raw_probability = _router_raw_predict(
                    router_x[disagreement], router_nodes, router_node_counts,
                    router_baseline,
                )
                probability = 1.0 / (1.0 + np.exp(-raw_probability))
                route = np.zeros(len(bank), dtype=bool)
                route[disagreement] = probability >= router_threshold
                return np.where(route, alternative_choice, current_choice).astype(int).tolist()
            if model_kind == "expression_ensemble_v12":
                decision = (
                    (1.0 - expression_fraction) * decision
                    + expression_fraction * expression_decision
                )
            combined = grouped.copy()
            combined.reshape(-1)[rows] += kernel_blend * decision
            if model_kind in ("listwise_trace_v23", "listwise_route_v24", "listwise_consensus_v25", "expert_consensus_v27", "listwise_capacity_v28", "program_search_v32"):
                candidate_trace = trace_execution_features(bank)
                equal, _, _ = _group_structure(bank)
                trace_dimension = candidate_trace.shape[2]
                trace_group = np.zeros(
                    (len(bank), 4, 2 * trace_dimension), dtype=np.float32
                )
                for question_index in range(len(bank)):
                    for group_index in np.flatnonzero(representative[question_index]):
                        members = equal[question_index, group_index]
                        trace_group[question_index, group_index, :trace_dimension] = (
                            candidate_trace[question_index, members].mean(axis=0)
                        )
                        trace_group[question_index, group_index, trace_dimension:] = (
                            candidate_trace[question_index, members].max(axis=0)
                        )
                semantic_dimension = (
                    listwise_dimension
                    if model_kind in ("listwise_capacity_v28", "program_search_v32")
                    else 48
                )
                listwise_x = np.concatenate((
                    features[:, :, 12:107],
                    features[:, :, 126:126 + semantic_dimension],
                    features[:, :, 254:254 + semantic_dimension],
                    trace_group,
                ), axis=2)
                if model_kind == "program_search_v32":
                    candidate_program = program_search_features(bank)
                    program_dimension = candidate_program.shape[2]
                    program_group = np.zeros(
                        (len(bank), 4, 2 * program_dimension), dtype=np.float32
                    )
                    for question_index in range(len(bank)):
                        for group_index in np.flatnonzero(representative[question_index]):
                            members = equal[question_index, group_index]
                            program_group[
                                question_index, group_index, :program_dimension
                            ] = candidate_program[question_index, members].mean(axis=0)
                            program_group[
                                question_index, group_index, program_dimension:
                            ] = candidate_program[question_index, members].max(axis=0)
                    listwise_x = np.concatenate((listwise_x, program_group), axis=2)
                if model_kind in ("listwise_trace_v23", "listwise_route_v24",
                                   "listwise_capacity_v28", "program_search_v32"):
                    listwise_x = (
                        listwise_x - listwise_feature_mean
                    ) / listwise_feature_scale
                if model_kind == "listwise_trace_v23":
                    hidden = np.tanh(
                        listwise_x @ listwise_first_weights + listwise_first_bias
                    )
                    listwise_scores = (
                        hidden @ listwise_second_weights + listwise_second_bias
                    )
                    combined += listwise_blend * listwise_scores
                elif model_kind == "listwise_route_v24":
                    listwise_scores = listwise_x @ listwise_weights + listwise_bias
                    current_choice = combined.argmax(axis=1)
                    listwise_choice = np.where(
                        representative, listwise_scores, -np.inf
                    ).argmax(axis=1)
                    row_numbers = np.arange(len(bank))
                    advantage = (
                        listwise_scores[row_numbers, listwise_choice]
                        - listwise_scores[row_numbers, current_choice]
                    )
                    ordered = np.sort(
                        np.where(representative, combined, -np.inf), axis=1
                    )
                    margin = ordered[:, -1] - ordered[:, -2]
                    route = (
                        (listwise_choice != current_choice)
                        & (advantage >= listwise_advantage_threshold)
                        & (margin <= listwise_margin_threshold)
                    )
                    return np.where(
                        route, listwise_choice, current_choice
                    ).astype(int).tolist()
                elif model_kind == "listwise_capacity_v28":
                    listwise_scores = listwise_x @ listwise_weights + listwise_bias
                    combined += listwise_blend * listwise_scores
                elif model_kind == "program_search_v32":
                    hidden = np.tanh(
                        listwise_x @ listwise_first_weights + listwise_first_bias
                    )
                    listwise_scores = (
                        hidden @ listwise_second_weights + listwise_second_bias
                    )
                    combined += listwise_blend * listwise_scores
                elif model_kind == "listwise_consensus_v25":
                    view_x = (
                        listwise_x,
                        listwise_x[:, :, :191],
                        np.concatenate((listwise_x[:, :, :95],
                                        listwise_x[:, :, 191:]), axis=2),
                        listwise_x[:, :, 95:],
                    )
                    score_stack = []
                    for current_x, (mean, scale, weights, bias) in zip(
                            view_x, listwise_models):
                        view_score = ((current_x - mean) / scale) @ weights + bias
                        score_stack.append(
                            np.where(representative, view_score, -np.inf)
                        )
                    score_stack = np.stack(score_stack, axis=0)
                    choices = score_stack.argmax(axis=2)
                    current_choice = combined.argmax(axis=1)
                    consensus_choice = np.empty(len(bank), dtype=np.int64)
                    votes = np.empty(len(bank), dtype=np.int64)
                    confidence = np.zeros(len(bank), dtype=np.float32)
                    for question_index in range(len(bank)):
                        counts = np.bincount(
                            choices[:, question_index], minlength=4
                        )
                        winner = int(counts.argmax())
                        consensus_choice[question_index] = winner
                        votes[question_index] = counts[winner]
                        agreeing = choices[:, question_index] == winner
                        confidence[question_index] = np.median(
                            score_stack[agreeing, question_index, winner]
                            - score_stack[
                                agreeing, question_index,
                                current_choice[question_index]
                            ]
                        )
                    ordered = np.sort(
                        np.where(representative, combined, -np.inf), axis=1
                    )
                    margin = ordered[:, -1] - ordered[:, -2]
                    route = (
                        (consensus_choice != current_choice)
                        & (votes >= listwise_minimum_votes)
                        & (confidence >= listwise_confidence_threshold)
                        & (margin <= listwise_margin_threshold)
                    )
                    return np.where(
                        route, consensus_choice, current_choice
                    ).astype(int).tolist()
                else:
                    current_choice = combined.argmax(axis=1)
                    row_numbers = np.arange(len(bank))
                    ordered = np.sort(
                        np.where(representative, combined, -np.inf), axis=1
                    )
                    margin = ordered[:, -1] - ordered[:, -2]

                    normalized23 = (listwise_x - v23_mean) / v23_scale
                    hidden23 = np.tanh(
                        normalized23 @ v23_first_weights + v23_first_bias
                    )
                    scores23 = hidden23 @ v23_second_weights + v23_second_bias
                    choice23 = (combined + v23_blend * scores23).argmax(axis=1)

                    normalized24 = (listwise_x - v24_mean) / v24_scale
                    scores24 = normalized24 @ v24_weights + v24_bias
                    alternative24 = np.where(
                        representative, scores24, -np.inf
                    ).argmax(axis=1)
                    advantage24 = (
                        scores24[row_numbers, alternative24]
                        - scores24[row_numbers, current_choice]
                    )
                    route24 = (
                        (alternative24 != current_choice)
                        & (advantage24 >= v24_advantage_threshold)
                        & (margin <= v24_margin_threshold)
                    )
                    choice24 = np.where(route24, alternative24, current_choice)

                    view_x = (
                        listwise_x,
                        listwise_x[:, :, :191],
                        np.concatenate((listwise_x[:, :, :95],
                                        listwise_x[:, :, 191:]), axis=2),
                        listwise_x[:, :, 95:],
                    )
                    score_stack = []
                    for current_x, (mean, scale, weights, bias) in zip(
                            view_x, v25_models):
                        view_score = ((current_x - mean) / scale) @ weights + bias
                        score_stack.append(
                            np.where(representative, view_score, -np.inf)
                        )
                    score_stack = np.stack(score_stack, axis=0)
                    view_choices = score_stack.argmax(axis=2)
                    alternative25 = np.empty(len(bank), dtype=np.int64)
                    votes25 = np.empty(len(bank), dtype=np.int64)
                    confidence25 = np.zeros(len(bank), dtype=np.float32)
                    for question_index in range(len(bank)):
                        counts = np.bincount(
                            view_choices[:, question_index], minlength=4
                        )
                        winner = int(counts.argmax())
                        alternative25[question_index] = winner
                        votes25[question_index] = counts[winner]
                        agreeing = view_choices[:, question_index] == winner
                        confidence25[question_index] = np.median(
                            score_stack[agreeing, question_index, winner]
                            - score_stack[
                                agreeing, question_index,
                                current_choice[question_index]
                            ]
                        )
                    route25 = (
                        (alternative25 != current_choice)
                        & (votes25 >= v25_minimum_votes)
                        & (confidence25 >= v25_confidence_threshold)
                        & (margin <= v25_margin_threshold)
                    )
                    choice25 = np.where(route25, alternative25, current_choice)

                    model_choices = np.stack(
                        (choice20, choice23, choice24, choice25), axis=0
                    )
                    result = current_choice.copy()
                    for question_index in range(len(bank)):
                        alternatives = model_choices[:, question_index]
                        alternatives = alternatives[
                            alternatives != current_choice[question_index]
                        ]
                        counts = np.bincount(alternatives, minlength=4)
                        winner = int(counts.argmax())
                        if counts[winner] >= v27_minimum_votes:
                            result[question_index] = winner
                    return result.astype(int).tolist()
            if model_kind == "dual_objective_v10":
                question_index, left_index, right_index, pair_rows = [], [], [], []
                for i in range(len(bank)):
                    reps = np.flatnonzero(representative[i])
                    for a in range(len(reps)):
                        for b in range(a + 1, len(reps)):
                            question_index.append(i)
                            left_index.append(reps[a])
                            right_index.append(reps[b])
                            pair_rows.append(features[i, reps[a]] - features[i, reps[b]])
                if pair_rows:
                    pair_x = (np.asarray(pair_rows, dtype=np.float32) - pair_feature_mean) / pair_feature_scale
                    pair_query_norm = np.square(pair_x).sum(axis=1, keepdims=True)
                    pair_distance = np.maximum(
                        pair_query_norm + pair_support_norm[None, :]
                        - 2.0 * (pair_x @ pair_support_vectors.T),
                        0.0,
                    )
                    pair_decision = (
                        np.exp(-pair_gamma * pair_distance) @ pair_dual_coef
                        + pair_intercept
                    )
                    pair_scores = np.zeros_like(grouped)
                    np.add.at(pair_scores, (question_index, left_index), pair_decision)
                    np.add.at(pair_scores, (question_index, right_index), -pair_decision)
                    combined += pair_blend * pair_scores
            return combined.argmax(axis=1).astype(int).tolist()
        return choices_from_answer_groups(
            bank, scores, reducer="max", count_bonus=count_bonus
        )

    if model_kind == "nonlinear_arithmetic_fusion":
        embedding_x = select_matrix(blocks, "embedding")
        embedding_x, _, _ = normalize_features(
            embedding_x, embedding_mean, embedding_scale
        )
        surface_x = select_matrix(blocks, "surface")
        surface_x, _, _ = normalize_features(surface_x, surface_mean, surface_scale)
        base_scores = (
            embedding_x @ embedding_weights
            + surface_gamma * (surface_x @ surface_weights)
        ) / base_score_scale

        random_x = np.tanh(
            random_feature_scale
            * (blocks["embedding"].reshape(-1, 1024) @ random_projection)
        )
        random_x, _, _ = normalize_features(random_x, random_mean, random_scale)
        nonlinear_scores = random_x @ random_weights / random_score_scale

        arithmetic_x = select_matrix(blocks, "arithmetic")[:, arithmetic_columns]
        arithmetic_x, _, _ = normalize_features(
            arithmetic_x, arithmetic_mean, arithmetic_scale
        )
        arithmetic_scores = (
            arithmetic_x @ arithmetic_weights / arithmetic_score_scale
        )
        scores = (
            base_scores
            + nonlinear_gamma * nonlinear_scores
            + arithmetic_gamma * arithmetic_scores
        ).reshape(len(bank), 4)
        return choices_from_answer_groups(
            bank, scores, reducer="max", count_bonus=count_bonus
        )

    if model_kind == "late_fusion":
        embedding_x = select_matrix(blocks, "embedding")
        embedding_x, _, _ = normalize_features(
            embedding_x, embedding_mean, embedding_scale
        )
        surface_x = select_matrix(blocks, "surface")
        surface_x, _, _ = normalize_features(surface_x, surface_mean, surface_scale)
        scores = (
            embedding_x @ embedding_weights + fusion_gamma * (surface_x @ surface_weights)
        ).reshape(len(bank), 4)
        return choices_from_answer_groups(
            bank, scores, reducer="max", count_bonus=count_bonus
        )

    matrix = select_matrix(blocks, feature_set)
    matrix, _, _ = normalize_features(matrix, mean, scale)
    scores = (matrix @ weights).reshape(len(bank), 4)
    if selection_mode == "answer_group":
        return choices_from_answer_groups(bank, scores, temperature=temperature)
    if selection_mode != "candidate":
        raise ValueError(f"Unknown checkpoint selection mode: {selection_mode}")
    return choices_from_scores(scores, blocks["valid"])
