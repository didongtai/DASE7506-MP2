"""Export the selected PCA/RBF V9 verifier as a NumPy-only checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import DATA, load_inputs
from explore_v8_kernel_router import prepare
from explore_v9_pca_kernel import group_projected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("checkpoint/model.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/v9-pca-kernel/model.npz"))
    args = parser.parse_args()

    tx, _, tl, _, tr, _, th, _, _, _, _ = prepare("")
    tb, ta = load_inputs("train", DATA)
    relative = ta["embeddings"] - ta["embeddings"].mean(axis=1, keepdims=True)
    relative /= np.maximum(np.linalg.norm(relative, axis=2, keepdims=True), 1e-6)
    pca = PCA(n_components=128, svd_solver="randomized", iterated_power=3, random_state=1)
    projected = pca.fit_transform(relative.reshape(-1, 1024)).astype(np.float32)
    tx = group_projected(tb, projected.reshape(len(tb), 4, -1), tx, tr)

    rows = (tr & th[:, None]).reshape(-1)
    x = tx.reshape(-1, tx.shape[2])[rows]
    y = tl.reshape(-1)[rows]
    model = make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", class_weight="balanced", cache_size=1024))
    model.fit(x, y)
    scaler, svc = model[0], model[1]

    with np.load(args.base, allow_pickle=False) as base:
        # Remove an earlier router if the current frozen checkpoint is used as base.
        payload = {name: base[name] for name in base.files
                   if not name.startswith("kernel_") and not name.startswith("semantic_pca_")}
    payload.update(
        model_kind=np.asarray("kernel_router_v9"),
        semantic_pca_mean=pca.mean_.astype(np.float32),
        semantic_pca_components=pca.components_.astype(np.float32),
        kernel_feature_mean=scaler.mean_.astype(np.float32),
        kernel_feature_scale=scaler.scale_.astype(np.float32),
        kernel_support_vectors=svc.support_vectors_.astype(np.float32),
        kernel_dual_coef=svc.dual_coef_.reshape(-1).astype(np.float32),
        kernel_intercept=np.asarray(svc.intercept_[0], dtype=np.float32),
        kernel_gamma=np.asarray(svc._gamma, dtype=np.float32),
        kernel_blend=np.asarray(1.5, dtype=np.float32),
        kernel_support_norm=np.square(svc.support_vectors_).sum(axis=1).astype(np.float32),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print({"path": str(args.output), "bytes": args.output.stat().st_size,
           "features": x.shape[1], "support_vectors": int(svc.support_.size),
           "gamma": float(svc._gamma),
           "pca_explained_variance": float(pca.explained_variance_ratio_.sum())})


if __name__ == "__main__":
    main()
