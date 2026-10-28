"""Bounded arithmetic-program search over question quantities."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.decomposition import PCA

from common import DATA, load_inputs, parse_answer
from explore_v5_numeric import NUMBER_RE
from explore_v7_confidence_routing import accuracy
from explore_v8_kernel_router import predict_kernel, prepare
from explore_v9_pca_kernel import group_projected
from explore_v11_relation_kernel import group_block
from explore_v22_trace_executor import trace_features
from explore_v23_listwise_ranker import predict_listwise
from explore_v28_listwise_semantic_capacity import features as base_features


def key(value):
    return round(float(value), 8)


def signed_log(value):
    return math.copysign(math.log1p(abs(value)), value)


def question_numbers(text):
    output = []
    for token in NUMBER_RE.findall(text):
        try:
            value = float(Fraction(token.replace(",", "")))
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if math.isfinite(value):
            output.append(value)
    return output[:8]


def bounded_programs(numbers, targets, beam=60):
    n = len(numbers)
    maximum_size = min(n, 5)
    tables = {}
    for index, number in enumerate(numbers):
        tables[1 << index] = {key(number): 0}
    target_logs = [signed_log(value) for value in targets if value is not None]
    hits = {key(value): [] for value in targets if value is not None}

    for size in range(1, maximum_size + 1):
        for mask in range(1, 1 << n):
            if mask.bit_count() != size:
                continue
            if size > 1:
                values = {}
                sub = (mask - 1) & mask
                while sub:
                    other = mask ^ sub
                    if other and sub < other and sub in tables and other in tables:
                        for left, left_ops in tables[sub].items():
                            for right, right_ops in tables[other].items():
                                candidates = (
                                    (left + right, 1), (left - right, 2),
                                    (right - left, 2), (left * right, 4),
                                )
                                for value, operation in candidates:
                                    if math.isfinite(value) and abs(value) <= 1e12:
                                        k = key(value)
                                        values[k] = values.get(k, 0) | left_ops | right_ops | operation
                                if abs(right) > 1e-10:
                                    value = left / right
                                    if math.isfinite(value) and abs(value) <= 1e12:
                                        k = key(value)
                                        values[k] = values.get(k, 0) | left_ops | right_ops | 8
                                if abs(left) > 1e-10:
                                    value = right / left
                                    if math.isfinite(value) and abs(value) <= 1e12:
                                        k = key(value)
                                        values[k] = values.get(k, 0) | left_ops | right_ops | 8
                    sub = (sub - 1) & mask
                if len(values) > beam:
                    def priority(item):
                        value = item[0]
                        distance = min((abs(signed_log(value) - target)
                                        for target in target_logs), default=0.0)
                        integer_bonus = -.05 if abs(value - round(value)) < 1e-8 else 0.0
                        return distance + integer_bonus
                    values = dict(sorted(values.items(), key=priority)[:beam])
                tables[mask] = values
            for target_key in hits:
                if target_key in tables.get(mask, {}):
                    hits[target_key].append((size, tables[mask][target_key], mask))
    all_values = [value for table in tables.values() for value in table]
    return hits, all_values


def program_features_for_question(row):
    numbers = question_numbers(row["question"])
    answers = [parse_answer(candidate["text"]) for candidate in row["candidates"]]
    targets = []
    for answer in answers:
        try:
            targets.append(float(Fraction(answer)) if answer is not None else None)
        except (ValueError, ZeroDivisionError, OverflowError):
            targets.append(None)
    hits, reachable = bounded_programs(numbers, targets)
    n = len(numbers)
    full_mask = (1 << n) - 1
    rows = []
    for target in targets:
        feature = np.zeros(19, dtype=np.float32)
        feature[1] = n / 8.0
        if target is None:
            rows.append(feature)
            continue
        feature[0] = 1.0
        target_hits = hits.get(key(target), [])
        feature[2] = float(any(abs(target - value) < 1e-8 for value in numbers))
        feature[3] = float(bool(target_hits))
        if target_hits:
            sizes = [item[0] for item in target_hits]
            feature[4] = max(sizes) / max(1, n)
            feature[5] = min(sizes) / max(1, n)
            feature[6] = math.log1p(len(target_hits)) / 4.0
            feature[7] = float(any(item[2] == full_mask for item in target_hits))
            for size in range(1, 6):
                feature[7 + size] = float(size in sizes)
            best = max(target_hits, key=lambda item: (item[0], -item[2]))
            for bit in range(4):
                feature[13 + bit] = float(bool(best[1] & (1 << bit)))
        target_log = signed_log(target)
        feature[17] = min((abs(target_log - signed_log(value))
                           for value in reachable), default=20.0) / 10.0
        feature[18] = float(abs(target - round(target)) < 1e-8)
        rows.append(feature)
    return np.stack(rows)


def program_features(bank):
    rows = []
    for index, row in enumerate(bank):
        rows.append(program_features_for_question(row))
        if (index + 1) % 500 == 0:
            print("program features", index + 1, "/", len(bank), flush=True)
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--salt", default="")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/v32-program-search.json"))
    parser.add_argument("--program-cache", type=Path,
                        default=Path("experiments/v32-program-cache.npz"))
    args = parser.parse_args()
    tx, vx, tl, vl, tr, vr, th, vh, folds, bt, bv = prepare(args.salt)
    tb, ta = load_inputs("train", DATA)
    vb, va = load_inputs("validation", DATA)
    rt = ta["embeddings"] - ta["embeddings"].mean(axis=1, keepdims=True)
    rv = va["embeddings"] - va["embeddings"].mean(axis=1, keepdims=True)
    rt /= np.maximum(np.linalg.norm(rt, axis=2, keepdims=True), 1e-6)
    rv /= np.maximum(np.linalg.norm(rv, axis=2, keepdims=True), 1e-6)
    pca = PCA(n_components=128, svd_solver="randomized", iterated_power=3,
              random_state=1)
    pt = pca.fit_transform(rt.reshape(-1, 1024)).astype(np.float32)
    pv = pca.transform(rv.reshape(-1, 1024)).astype(np.float32)
    gt = group_projected(tb, pt.reshape(len(tb), 4, -1), tx, tr)
    gv = group_projected(vb, pv.reshape(len(vb), 4, -1), vx, vr)
    trace_t = group_block(tb, trace_features(tb), tr)
    trace_v = group_block(vb, trace_features(vb), vr)
    base_t = base_features(gt, trace_t, 16)
    base_v = base_features(gv, trace_v, 16)
    if args.program_cache.exists():
        with np.load(args.program_cache) as cache:
            raw_program_t = cache["train"]
            raw_program_v = cache["validation"]
    else:
        raw_program_t = program_features(tb)
        raw_program_v = program_features(vb)
        args.program_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.program_cache, train=raw_program_t, validation=raw_program_v)
    program_t = group_block(tb, raw_program_t, tr)
    program_v = group_block(vb, raw_program_v, vr)

    ko, kv, _, _, _ = predict_kernel(1.0, "scale", gt, gv, tl, tr, vr, th, folds)
    ko -= ko.max(axis=1, keepdims=True)
    kv -= kv.max(axis=1, keepdims=True)
    v9t, v9v = bt + 1.5 * ko, bv + 1.5 * kv
    results = []
    for use_program in (False, True):
        xt = np.concatenate((base_t, program_t), axis=2) if use_program else base_t
        xv = np.concatenate((base_v, program_v), axis=2) if use_program else base_v
        for config in ((0, 300, .01, 1e-4), (12, 160, .005, 2e-3)):
            lt, lv = predict_listwise(xt, xv, tl, tr, vr, th, folds, config)
            lt -= lt.max(axis=1, keepdims=True)
            lv -= lv.max(axis=1, keepdims=True)
            for blend in (.2, .5, .75, 1.0, 1.5, 2.0):
                pred_t = (v9t + blend * lt).argmax(1)
                pred_v = (v9v + blend * lv).argmax(1)
                results.append({
                    "program_search": use_program,
                    "feature_count": int(xt.shape[2]), "hidden": config[0],
                    "blend": blend,
                    "cv_informative_correct": accuracy(pred_t, tl, th),
                    "validation_informative_correct": accuracy(pred_v, vl, vh),
                })
    results.sort(key=lambda row: (
        row["cv_informative_correct"] + row["validation_informative_correct"],
        row["validation_informative_correct"], row["cv_informative_correct"],
    ), reverse=True)
    record = {"salt": args.salt, "test_used": False,
              "candidate_program_features": 19,
              "baseline_v9": {"cv_informative_correct": accuracy(v9t.argmax(1), tl, th),
                              "validation_informative_correct": accuracy(v9v.argmax(1), vl, vh)},
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": record["baseline_v9"],
                      "best": results[:30]}, indent=2))


if __name__ == "__main__":
    main()
