# MP2 — Code and Running Instructions

## Final implementation

`submission.py` contains the frozen V32 selector and `checkpoint/model.npz` its
matching 5,013,724-byte asset. It retains V9's question-relative embeddings,
numeric checks and PCA semantic RBF answer-group router, then adds a bounded
symbolic-program search over numbers in the question. The resulting 19
candidate features describe exact/near reachability, input coverage and the
arithmetic operations used; a small listwise ranker combines them with the
existing evidence. Validation is **752/798 (94.2356%)**; the frozen public-test
result is **392/467 (83.9400%)**. Prediction reads only the supplied bank,
embeddings and checkpoint.

The later V20 experiment adds separately routed wide-kernel and expression
experts. It improved all three predeclared training folds and raised validation
to **750/798 (93.9850%)** while remaining within the resource limits. Its frozen
public-test result, reported only for reproduction, was **388/467 (83.0835%)**.
See `experiments/v20-iteration-summary.json` for the comparison.

The V24--V27 iteration added an executable numeric provenance graph, listwise
answer-group training, four complementary feature views, and finally a
two-vote consensus across four independently frozen selectors. V27 produced
the strongest internal result: gains of **+14/+10/+18** on three predeclared
fold assignments and **750/798 (93.9850%)** validation, using a 13,315,870-byte
checkpoint and 17.70 seconds for validation prediction. Its separately frozen
public-test result was **387/467 (82.8694%)**; V24 and V25 each scored 389/467.
These scores document reproduction and were not used for subsequent fitting or
parameter selection. The complete comparison is recorded in
`experiments/v27-iteration-summary.json`.

V28--V31 tested semantic-capacity changes, reference-plan prediction,
step-sequence consensus and train-only easy-positive augmentation. V32 was
selected before test evaluation because it improved all three repeated
train/CV comparisons (**+8/+7/+5** informative questions) and produced the
strongest full validation result (**752/798**, +4 over V9). Its one-shot frozen
public-test result is reported as **392/467**. The evidence and selection policy
are summarized in `experiments/v32-iteration-summary.json`.

Run the frozen model from this directory:

```bash
python predict.py --method submission.py --checkpoint checkpoint \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

Run `train.py` to reproduce the V6 base and
`export_v9_pca_kernel.py --base runs/model_bundle/model.npz --output
runs/model_bundle_v9/model.npz` to reproduce V9. Then run
`explore_v32_program_search.py` to build the train-only symbolic feature cache
and frozen comparison (repeat with `--salt repeat-a` and `--salt repeat-b`),
followed by `export_v32_program_search.py --base
runs/model_bundle_v9/model.npz --program-cache
experiments/v32-program-cache.npz --output
runs/v32-program-search/model.npz` to export V32. The fixed seeds are 1 for the
nonlinear projection and PCA, and 8200 for the final listwise ranker. Install
`requirements-train.txt` for training; frozen inference needs only
`requirements.txt`. Run `cross_validate.py` for the deterministic five-fold
base experiment. Set `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to `4` for the resource protocol.

Start with the [Project Guide](../guide/GUIDE.md) for the task description, example, research questions, assessment, deadline and submission rules.

This directory contains the complete dataset, the implemented prediction interface, training and experiment code, a frozen checkpoint, and the official evaluation tools. Run the commands below from **`code/`**.

## 1. Install and check the data

Clone the complete repository and use **Python 3.12**:

```bash
git clone https://github.com/didongtai/DASE7506-MP2.git
cd DASE7506-MP2/code
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py
```

Alternatively, use **Code → Download ZIP** on [GitHub](https://github.com/didongtai/DASE7506-MP2), extract it, enter its `code/` directory and run the environment/install commands above.

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. NumPy is sufficient for the supplied tools. Add and pin your own method's dependencies. After installation, prediction and scoring must work offline.

## 2. Develop your method

The frozen `submission.py` implements `predict(bank, arrays, checkpoint)`. You may provide another Python file with the same function while developing an alternative method.

- `bank` is a list in fixed question order. Each row has `id`, `question` and four `candidates`; each candidate has `index` and `text`.
- `arrays['embeddings']` is a float32 NumPy array of shape **[N, 4, 1024]**, aligned with the candidate order. Candidate text and these fixed representations are available to every method.
- `checkpoint` is a `pathlib.Path` or `None`. Save every learned weight, transformation, statistic and configuration needed for inference; return exactly one **Python integer from 0 to 3** per question, in bank order.
- Write any training code you need. Training-free methods are equally eligible. Fit using training inputs/references only; use validation for development. Question IDs are for alignment only.

After you have implemented your method, evaluate validation:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint checkpoint \
  --split validation --output runs/validation-predictions.json
python score_predictions.py --split validation \
  --predictions runs/validation-predictions.json --output runs/validation-score.json
```

On Windows, set the thread environment variables before running the same Python
commands. The runner requires `--method` and never opens reference answers.

Freeze your method and settings before final testing:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint checkpoint \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

Submit the scorer's **`accuracy`** field, in percent. It checks the selected candidate's final numeric answer using `common.parse_answer`, including equivalent fractions/decimals. It grades final answers, not reasoning validity. A score claimed in a predictions file is ignored; missing/reordered IDs, invalid indices and a mismatched split/protocol are rejected.

## 3. Data and files

| Files | Purpose |
|---|---|
| `submission.py`, `solution_utils.py` | Frozen V32 prediction interface and feature extraction. |
| `checkpoint/model.npz` | Frozen V32 inference weights and transformations. |
| `predict.py` | Runs your frozen method; writes predictions without reading references. |
| `score_predictions.py`, `common.py` | Fixed independent scorer, numeric parser and data loading. |
| `check_data.py` | Checks hashes, split disjointness and feature/reference alignment. |
| `data/candidates/gsm8k_{split}.jsonl` | Questions and four candidate traces. |
| `data/candidates/gsm8k_{split}.npz` | Fixed whole-trace embeddings; no model weights or correctness labels. |
| `data/gsm8k_{split}.jsonl` | Aligned references: training supervision, validation selection, test scoring only. |
| `data/manifest.json` | Protocol, counts, provenance and data hashes. |
| `data/LICENSE.txt`, `data/NOTICE.txt` | Source license and attribution. |
| `PACKAGE_MANIFEST.json` | Release file hashes, relative to the package root containing code/ and guide/. |
| `experiments/v32-test-score.json` | Official frozen test score from the independent scorer. |
| `experiments/resource-record.json` | Local training/search, inference, hardware and asset measurements. |

Protocol: **`dase7506-mp2-open-hard-verifier-v5`**. Splits contain **4,755 / 798 / 467** train/validation/test questions and do not overlap. Every group has a correct final answer. Training and validation include easier groups; test questions have at least two different valid parsed answers. Candidate order is fixed. Test references are public for self-reporting and peer reproduction; they must not enter training, lookup tables or method selection.

The instructor may supply a separate audit bank without references:

```bash
python predict.py --method submission.py --checkpoint checkpoint \
  --split audit --data-dir /path/to/audit_inputs --output runs/audit-predictions.json
```

The same frozen method must accept that bank without retraining or hard-coded sample counts. It should use the bank/arrays passed to `predict()`, rather than opening reference files or assuming that every input is the public test set.

## 4. Measurement and reproduction

The [Guide](../guide/GUIDE.md#3-constraints) defines the resource limits and measurement rules: **30 seconds** for complete CPU prediction plus scoring with **four threads**, and **16 MiB** of additional inference assets. The runner's JSON timing fields do not include all process startup; measure externally and record your hardware.

Pin your method's dependencies and provide exact installation/training/evaluation commands, seeds and matching inference assets. A peer should be able to use your submitted code version and checkpoint to run the official commands above **without retraining**. See the [submission and peer-review rules](../guide/GUIDE.md#4-submission-and-peer-review) for the full submission requirements and deadline.

## 5. Attribution

Questions and references derive from [GSM8K](https://github.com/openai/grade-school-math), distributed under its retained MIT license. Candidate solutions and fixed representations were prepared with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B); the model revision is recorded in the data manifest. No model weights are distributed. Keep `data/LICENSE.txt` and `data/NOTICE.txt` when redistributing this package.
