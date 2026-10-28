# Selecting Reliable LLM Reasoning with a Neuro-Symbolic Trace Verifier

**DASE7506 Advanced Machine Learning Methods - Mini-project 2**

## 1. Research questions and hypotheses

I investigate three questions from the brief:

1. **Does reading the reasoning help?** Supplied whole-trace embeddings should
   outperform final-answer agreement and shallow text features.
2. **Should a solution be judged comparatively?** Question-relative embeddings
   and answer-group evidence should outperform independent ranking.
3. **When does more computation pay off?** Compact nonlinear and symbolic
   checks should add complementary evidence, while unconstrained model capacity
   may increase variance.

Training references fit model parameters; validation supports development and
selection. V32 was selected from repeated training-only comparisons and
validation evidence. Its code, constants, and checkpoint were frozen before
the single reported public-test run. Test questions, references, and
per-example outcomes never enter fitting, feature construction, or parameter
selection.

## 2. Data and metric

The supplied bank contains 4,755 training, 798 validation, and 467 test
questions. Each has four candidate traces and one supplied 1,024-dimensional
representation per trace. The required output is one integer candidate index.
The scorer parses the selected final numeric answer and compares it with the
reference; it does not require every intermediate reasoning step to be valid.

Validation contains 379 unanimous-answer groups and 419 disagreement groups.
Choice cannot affect correctness in a valid unanimous group, so I report both
complete accuracy and disagreement-only accuracy. Every test group contains at
least two distinct valid parsed answers.

## 3. Method

### 3.1 Baselines and relative representation

The answer-only baseline groups scorer-compatible fractions and decimals and
selects the largest group. A 23-feature surface model uses validity, group size,
candidate position, length and number counts, question-number coverage, answer
magnitude, conclusion markers, and equation/operator counts.

For candidate embedding $e_{ij}$, I form a question-relative representation

$$
r_{ij}=\frac{e_{ij}-\frac14\sum_{k=1}^4e_{ik}}
{\left\|e_{ij}-\frac14\sum_{k=1}^4e_{ik}\right\|_2}.
$$

Training-only means and standard deviations normalize each dimension. Ridge
regression predicts whether a candidate's parsed answer equals the reference.
The embedding and surface predictions form the scaled base score
$b_j=\operatorname{scale}(s_j^{(e)}+0.25s_j^{(u)})$.

### 3.2 Complementary verifiers and semantic routing

A fixed Gaussian projection with seed 1 maps each relative embedding to 128
nonlinear features, $z_{ij}=\tanh(0.5r_{ij}R)$, followed by ridge regression.
Two safe parsers inspect explicit numeric equality chains. The strict version
accepts only direct arithmetic, while the tolerant version handles common
LaTeX fractions, decimals, approximation signs, and unit suffixes. Both use an
arithmetic AST whitelist and never execute candidate text.

A 31-feature numeric block describes the four final answers jointly: valid
answer count, exact rank among unique answers, distance from the group median,
minimum/maximum status, sign, integrality, scale relative to question numbers,
and reachability by one pairwise operation. Ridge and logistic models score this
block. The frozen V6 candidate score is

$$
s_j=b_j+0.475s_j^{(n)}+0.1s_j^{(a1)}+0.25s_j^{(a2)}
    +0.5s_j^{(r)}+0.9s_j^{(l)}.
$$

Equal parsed answers are grouped using

$$
g(A)=\max_{j:a_j=A}s_j+0.275\log|\{j:a_j=A\}|.
$$

V9 adds a 128-dimensional PCA basis fitted to normalized training embeddings;
the retained components explain 87.96% of their variance. Each answer group
combines mean and maximum PCA features with the six V6 component scores and
structural features. A class-balanced RBF SVM with $C=1$ is trained on
training-only out-of-fold features and blended into V6 with weight 1.5. The PCA
basis and 2,547 support vectors are evaluated by NumPy at inference;
scikit-learn is only a training dependency.

### 3.3 Bounded symbolic program search and listwise fusion

V32 extracts up to eight numeric quantities from the question and constructs
bounded arithmetic programs using addition, subtraction, multiplication, and
division. Programs may consume at most five distinct input occurrences. For
each number subset, a target-aware beam retains at most 60 intermediate values,
keeping runtime predictable without using a question ID or reference answer.

For each candidate answer, 19 features summarize exact reachability, direct
occurrence in the question, minimum and maximum program size, full-input
coverage, the number of reachable programs, operation types, distance to the
closest reachable value, and answer integrality. Mean and maximum features are
pooled within equal-answer groups. They are concatenated with compact semantic
and executed-trace features and scored by a 12-unit tanh listwise ranker. The
final score is

$$
s_j^{(V32)}=s_j^{(V9)}+1.5s_j^{(program)}.
$$

The search does not try to reproduce a natural-language rationale. It asks the
narrower question of whether an answer is compatible with a short arithmetic
construction from quantities explicitly present in the problem.

### 3.4 Selection protocol

I compared candidate ranking, answer grouping, late fusion, listwise training,
random-feature ensembles, alternative targets, nearest-neighbour and lexical
models, operation closure, executed-trace provenance, stacking, wider kernels,
pairwise ranking, consensus selectors, reference-plan prediction, and
train-only easy-positive augmentation. New components were admitted only when
validation and repeated training-only cross-validation supported them.

V32 was selected before test evaluation because it produced the strongest full
validation result and improved V9 on informative questions under all three
repeated train/CV partitions. Its hidden size, blend weight, search limits, and
checkpoint were frozen before public-test scoring.

## 4. Results

| Method | Validation correct | Validation accuracy | Disagreement correct | Disagreement accuracy |
|---|---:|---:|---:|---:|
| First candidate | 624/798 | 78.20% | 245/419 | 58.47% |
| Answer majority | 688/798 | 86.22% | 309/419 | 73.75% |
| Surface features | 706/798 | 88.47% | 327/419 | 78.04% |
| Candidate embedding | 727/798 | 91.10% | 348/419 | 83.05% |
| Answer-group embedding | 732/798 | 91.73% | 353/419 | 84.25% |
| Late fusion (V3) | 736/798 | 92.23% | 357/419 | 85.20% |
| Nonlinear embedding fusion | 740/798 | 92.73% | 361/419 | 86.16% |
| Nonlinear + strict arithmetic (V4) | 742/798 | 92.98% | 363/419 | 86.63% |
| + tolerant arithmetic + numeric ridge (V5) | 746/798 | 93.48% | 367/419 | 87.59% |
| + logistic numeric calibration (V6) | 746/798 | 93.48% | 367/419 | 87.59% |
| + random-projection RBF router (V8) | 747/798 | 93.61% | 368/419 | 87.83% |
| + PCA semantic RBF router (V9) | 748/798 | 93.73% | 369/419 | 88.07% |
| + bounded program search + listwise ranker (V32) | **752/798** | **94.24%** | **373/419** | **89.02%** |

### 4.1 Controlled ablation study

The following matched comparisons isolate key design choices. Positive deltas
mean that restoring the named component improves over the corresponding
ablated method; this is more informative than treating the entire development
sequence as a single comparison. Every pair uses the same frozen validation
split and scorer. Because all 379 unanimous-answer groups are correct under any
selection, the change in full-validation correct count is exactly the change
on the 419 disagreement questions; disagreement accuracy therefore gives the
more sensitive view of each component.

| Design choice restored | Ablated method | Method with component | Validation delta | Disagreement delta |
|---|---|---|---:|---:|
| Full-trace reasoning representation | Surface features: 706/798 | Candidate embedding: 727/798 | +21 (+2.63 pp) | +21 (+5.01 pp) |
| Cross-candidate answer grouping | Candidate embedding: 727/798 | Answer-group embedding: 732/798 | +5 (+0.63 pp) | +5 (+1.19 pp) |
| Explicit arithmetic checking | Nonlinear embedding: 740/798 | V4: 742/798 | +2 (+0.25 pp) | +2 (+0.48 pp) |
| Logistic numeric calibration | V5: 746/798 | V6: 746/798 | 0 (0.00 pp) | 0 (0.00 pp) |
| Bounded symbolic module | V9: 748/798 | V32: 752/798 | +4 (+0.50 pp) | +4 (+0.95 pp) |

**Reasoning representation versus surface cues.** Replacing the 23 shallow
features with the supplied full-trace representation adds 21 correct choices.
On disagreement questions, errors fall from 92 to 71, a 22.8% relative error
reduction. Length, formatting, equation counts, and confident conclusion
markers therefore do not explain most of the learned verifier's advantage.
This comparison does not prove that every embedding dimension represents valid
reasoning: the fixed representation may still encode unmeasured stylistic
cues. It shows the narrower and defensible result that the supplied whole-trace
representation contains useful information beyond the audited surface block.

**Comparative judgement.** Pooling candidates by their parsed answer raises
the result from 727 to 732 correct. All five recovered decisions occur in the
disagreement subset, reducing its errors from 71 to 66 (7.0%). The gain is
small but aligned with the hypothesis: a candidate is easier to judge when
agreement with independently written alternatives is visible. Grouping cannot
help unanimous groups and can still reinforce a shared mistake, so it is used
as corroboration rather than as a correctness rule.

**Arithmetic evidence and model capacity.** Adding the strict arithmetic
checker to nonlinear embeddings recovers two questions, reducing disagreement
errors from 58 to 56 (3.4%). Its modest coverage is expected because it can
verify only equations explicitly written in the trace; it cannot detect a
misread condition that is followed by internally consistent arithmetic. By
contrast, adding logistic numeric calibration in V6 changes no top-1 decision.
This zero result is useful negative evidence: additional nonlinear capacity is
not automatically beneficial. The logistic score can still differ and be used
by later fusion, but the experiment does not credit it with an independent
accuracy gain.

**Bounded symbolic module.** Removing program-search features and their
12-unit listwise scorer gives V9; restoring the combined module gives V32.
V32 changes 11 selected answers: seven repair a V9 error, three introduce an
error, and one leaves correctness unchanged, for a net gain of four.
Disagreement errors fall from 50 to 46, an 8.0% relative reduction. The three
regressions show that arithmetic reachability is evidence, not proof.

The single-split gain is too small to stand alone. Under the default, salt-A,
and salt-B training/CV partitions, V32 improves informative CV accuracy over
V9 by 8, 7, and 5 questions; the corresponding informative-validation gains
are 4, 6, and 5. The direction is consistent across all three runs, which is
stronger evidence than the +4 headline alone, although repeated use of the
same validation bank still leaves some model-selection bias.

**Accuracy-cost trade-off and scope.** Fresh program-feature search takes
257.07 seconds, 84.0% of the 306.11-second clean reproduction process. The
resulting checkpoint is 4.78 MiB, and complete prediction plus independent
scoring takes 18.23 seconds locally. The symbolic module therefore has a modest
accuracy return for most of the reproduction-time computation, but remains
worth retaining because its repeated gains are positive and the deployed
method stays inside the 16 MiB and 30-second limits. This is a module-level
ablation: search features and their learned fusion are removed together. It
does not identify how much of the gain comes from the search itself versus the
listwise ranker, and no such finer-grained claim is made.

Early training-only robustness results were:

| Method | Default folds | Salt A | Salt B |
|---|---:|---:|---:|
| V4 | 4376/4755 | 4360/4755 | 4361/4755 |
| V5 | 4381/4755 | 4386/4755 | 4380/4755 |
| V6 | 4388/4755 | 4390/4755 | 4380/4755 |
| V8 | 4389/4755 | 4390/4755 | 4382/4755 |
| V9 | 4396/4755 | 4394/4755 | 4382/4755 |

For the later program-search comparison, the frozen hidden-12, blend-1.5 model
improves V9 on informative train/CV questions under the default, salt-A, and
salt-B partitions by 8, 7, and 5 questions. Its corresponding informative
validation gains are 4, 6, and 5. This repeated internal evidence determined
the final configuration.

Frozen aggregate test results were V1 369/467, V2 374/467, V3 372/467, V4
378/467, V5 386/467, V6 386/467, V8 386/467, V9 390/467, and V32
**392/467 = 83.940043%**. These outcomes document frozen iterations; test
questions, references, and per-example outcomes do not enter training or
parameter selection.

## 5. Resource use and reproducibility

Training uses fixed seeds 1 (random projection and PCA) and 8200 (listwise
ranker). On an AMD Ryzen 9 8945HX, a clean base train, V9 export, fresh V32
search/cache build, and V32 export take 17.84, 13.20, 257.07, and 19.00 seconds
respectively (306.11 seconds total). The regenerated model produces identical
validation and test choices. With `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`MKL_NUM_THREADS` fixed to four, fresh V32 prediction followed by independent
scoring takes 18.23 seconds. The reference Xeon remains authoritative.

The only inference asset is `model.npz`, 5,013,724 bytes (4.78 MiB), SHA-256
`b384830a79de9d98e204d4273987b0ea7afaf0e84c008bdb0758a06d610af870`.
It is below the 16 MiB limit. Inference uses NumPy, has no network access,
performs no generation or retraining, and contains no ID lookup or fixed sample
count.

## 6. Limitations and error analysis

Whole-trace embeddings can rank solutions but cannot identify a causal error
step. Equation checks detect explicit inconsistencies, yet a self-consistent
trace may misread the question or omit a condition. The bounded search is also
intentionally incomplete: it uses only the first eight parsed quantities, at
most five inputs, four elementary operations, rounded numeric states, and a
finite beam. It does not directly model units, implicit constants, rates
expressed only in language, or longer multi-stage programs.

Despite these limitations, symbolic reachability is more interpretable than
another unrestricted score-fusion layer and produced consistent internal
gains. Test accuracy remains materially below validation disagreement accuracy,
showing distribution shift. Future work should add unit- and relation-aware
symbolic states while retaining bounded, reference-free inference.

## 7. Conclusion

Reasoning representations provide the largest gain over answer agreement.
Comparing candidates through relative embeddings and answer groups adds further
value. Compact nonlinear, arithmetic, and numeric checks help, and bounded
symbolic search provides another reliable gain by testing whether answers can
be constructed from problem quantities. The final reproducible V32 selector
reaches 94.24% validation and 83.94% frozen public-test accuracy within the CPU
and asset limits.

## AI assistance and reuse disclosure

OpenAI Codex provided substantive assistance with specification review, feature
and experiment implementation, debugging, evaluation, and report drafting. The
submitter is responsible for understanding and verifying the work. Frozen
inference uses NumPy, while training additionally uses scikit-learn. No external
dataset, pretrained weight, or third-party solution is used.
