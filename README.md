# DASE7506 MP2 - Neuro-Symbolic Trace Verifier

This repository contains a reproducible solution for **MP2: Selecting Reliable
LLM Reasoning**. The method learns compact verifiers from the supplied training
split, scores the four fixed candidate solutions using linear and nonlinear
projections of their whole-trace embeddings, checks explicit arithmetic,
models the numeric structure of candidate answers, aggregates evidence from
candidates with the same parsed answer, and performs a bounded arithmetic
program search over quantities in each question.
It performs no generation, uses no external data or pretrained
weights, and reads no reference answers during prediction.

## Result

| Method | Validation accuracy | Disagreement-only validation accuracy |
|---|---:|---:|
| First candidate | 78.20% | 58.47% |
| Final-answer majority vote | 86.22% | 73.75% |
| Surface features | 88.47% | 78.04% |
| Candidate-level full-trace embeddings | 91.10% | 83.05% |
| Answer-group full-trace embeddings | 91.73% | 84.25% |
| Surface + embeddings | 90.98% | 82.82% |
| Late-fused answer-group verifier (V3) | 92.23% | 85.20% |
| Nonlinear embedding fusion | 92.73% | 86.16% |
| Nonlinear + arithmetic verifier (V4) | 92.98% | 86.63% |
| Structured ridge verifier (V5) | **93.48%** | **87.59%** |
| Ridge + logistic numeric verifier (V6) | 93.48% | 87.59% |
| Random-projection RBF answer-group router (V8) | 93.61% | 87.83% |
| PCA semantic RBF answer-group router (V9) | 93.73% | 88.07% |
| Bounded program search + listwise ranker (V32, final) | **94.24%** | **89.02%** |

The frozen V32 model obtains **83.940043% (392/467)** on the official test bank.
It was selected before test evaluation from three repeated training/CV
comparisons and validation evidence: the symbolic component improved V9 by
8, 7, and 5 informative CV questions and raised complete validation from
748/798 to 752/798. Every test question has disagreeing candidate answers, so the
disagreement-only validation result is the relevant development comparison.
The official frozen score is in `code/experiments/v32-test-score.json`; the
selection evidence is in `code/experiments/v32-iteration-summary.json`.

## Reproduce

Use Python 3.12.  From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code\requirements-train.txt
Set-Location code
python check_data.py
```

Train from the supplied training references and select the configuration on
validation:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
python train.py --output-dir runs\model_bundle
python export_v9_pca_kernel.py --base runs\model_bundle\model.npz `
  --output runs\model_bundle_v9\model.npz
python explore_v32_program_search.py `
  --output experiments\v32-program-search.json
python explore_v32_program_search.py --salt repeat-a `
  --output experiments\v32-program-search-a.json
python explore_v32_program_search.py --salt repeat-b `
  --output experiments\v32-program-search-b.json
python export_v32_program_search.py --base runs\model_bundle_v9\model.npz `
  --program-cache experiments\v32-program-cache.npz `
  --output runs\v32-program-search\model.npz
```

The fixed random seeds are 1 for the nonlinear projection and PCA, and 8200
for the final listwise ranker. On the recorded local machine, a clean base
train, V9 export, fresh V32 search/cache build and V32 export take approximately
17.84, 13.20, 257.07 and 19.00 seconds respectively (306.11 seconds total).
See `code/experiments/resource-record.json` for the exact environment and
measurements.

Reproduce the frozen test prediction and official score without retraining:

```powershell
python predict.py --method submission.py --checkpoint checkpoint `
  --split test --output runs\test-predictions.json
python score_predictions.py --split test `
  --predictions runs\test-predictions.json --output runs\test-score.json
```

The checked-in checkpoint is 5,013,724 bytes. Its SHA-256 is
`b384830a79de9d98e204d4273987b0ea7afaf0e84c008bdb0758a06d610af870`.

## Files

- `code/submission.py`: frozen prediction interface.
- `code/solution_utils.py`: scorer-compatible answer grouping and features.
- `code/train.py`: deterministic training, ablation, and selection procedure.
- `code/export_v9_pca_kernel.py`: fits and exports the PCA semantic RBF base.
- `code/explore_v32_program_search.py`: evaluates the bounded symbolic features.
- `code/export_v32_program_search.py`: exports the selected V32 listwise model.
- `code/cross_validate.py`: deterministic five-fold training-only robustness check.
- `code/checkpoint/model.npz`: matching inference asset.
- `code/experiments/`: validation, test, and resource records.
- `code/experiments/v32-test-score.json`: official frozen test score JSON.
- `report/report.md`: report source.
- `output/pdf/mp2-report.pdf`: rendered final report.

## Submission package

- Reported leaderboard accuracy: **83.940043**.
- Code version: the commit or release containing this README and
  `code/PACKAGE_MANIFEST.json`.
- Checkpoint/configuration link: `code/checkpoint/model.npz`.
- Official evaluation JSON: `code/experiments/v32-test-score.json`.
- Report: `output/pdf/mp2-report.pdf` (4 pages).

The remaining submission actions are external to this repository: publish the
chosen commit/release, enter the same student ID and score on the MP2 website,
then complete the generated GitHub issue before the deadline.

## Data boundary and disclosure

Only the supplied training inputs and references are used to fit parameters.
Validation references are used for configuration selection. Each reported
version was scored only after its configuration was frozen. V32 was chosen from
repeated training/CV and validation results, without using any test
question, reference, or per-example test outcome for training or tuning.
Question IDs are used only for alignment and deterministic fold assignment.

OpenAI Codex provided substantive assistance in reading the specification,
implementing and testing the method, running the documented experiments, and
drafting documentation. The final submitter is responsible for understanding,
checking, and being able to explain the implementation. Frozen inference uses
NumPy; training additionally uses scikit-learn. No external dataset,
pretrained weight, third-party solution, or additional model asset is reused.
