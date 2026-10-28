"""Export the frozen symbolic-program/listwise V32 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from common import DATA, load_inputs
from explore_v8_kernel_router import prepare
from explore_v9_pca_kernel import group_projected
from explore_v11_relation_kernel import group_block
from explore_v22_trace_executor import trace_features
from explore_v23_listwise_ranker import fit_ranker
from explore_v28_listwise_semantic_capacity import features as base_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("checkpoint/model.npz"))
    parser.add_argument("--program-cache", type=Path,
                        default=Path("experiments/v32-program-cache.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/v32-program-search/model.npz"))
    args = parser.parse_args()
    tx, _, tl, _, tr, _, th, _, _, _, _ = prepare("")
    bank, arrays = load_inputs("train", DATA)
    relative = arrays["embeddings"] - arrays["embeddings"].mean(axis=1, keepdims=True)
    relative /= np.maximum(np.linalg.norm(relative, axis=2, keepdims=True), 1e-6)
    pca = PCA(n_components=128, svd_solver="randomized", iterated_power=3,
              random_state=1)
    projected = pca.fit_transform(relative.reshape(-1, 1024)).astype(np.float32)
    group = group_projected(bank, projected.reshape(len(bank), 4, -1), tx, tr)
    trace = group_block(bank, trace_features(bank), tr)
    with np.load(args.program_cache) as cache:
        program = group_block(bank, cache["train"], tr)
    x = np.concatenate((base_features(group, trace, 16), program), axis=2)
    mean, scale, parameters = fit_ranker(
        x[th], tl[th], tr[th], hidden=12, epochs=160,
        learning_rate=.005, weight_decay=2e-3, seed=8200,
    )
    with np.load(args.base, allow_pickle=False) as base:
        payload = {name: base[name] for name in base.files}
    payload.update(
        model_kind=np.asarray("program_search_v32"),
        listwise_dimension=np.asarray(16, dtype=np.int32),
        listwise_blend=np.asarray(1.5, dtype=np.float32),
        listwise_feature_mean=mean.astype(np.float32),
        listwise_feature_scale=scale.astype(np.float32),
        listwise_first_weights=parameters[0].astype(np.float32),
        listwise_first_bias=parameters[1].astype(np.float32),
        listwise_second_weights=parameters[2].astype(np.float32),
        listwise_second_bias=parameters[3].astype(np.float32),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print({"path": str(args.output), "bytes": args.output.stat().st_size,
           "input_features": int(x.shape[2]), "hidden": 12, "blend": 1.5})


if __name__ == "__main__":
    main()
