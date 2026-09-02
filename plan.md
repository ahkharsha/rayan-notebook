# DG-Hetero-GNN Notebook Recovery Plan

## 1. Objective

Produce two executable and validated notebooks from the implementation found at repository commit 9 (`e74b052d2af03850bdad515d82fd5d3538387efe`):

1. `DG_Hetero_GNN_Paper_Minimal_Diff.ipynb`
   - Starts from commit 9.
   - Makes the smallest practical set of changes required to satisfy the paper.
   - Retains the original code organization and implementation style wherever they do not conflict with the paper.

2. `DG_Hetero_GNN_Paper_Clean_Reproduction.ipynb`
   - Begins as a copy of the corrected minimal-diff notebook.
   - Refactors the implementation into a clean, modular, auditable notebook.
   - Implements the same scientific protocol as the minimal-diff notebook.

The two notebooks must produce equivalent metrics and predictions when run with the same data, seed, configuration, software versions, and device-determinism settings.

## 2. Single Source of Truth

The sole normative specification is:

`D:\Rayan Paper materials\Preparation_of_Papers_for_IEEE_ACCESS (25).pdf`

The following rules are mandatory:

- The paper overrides commit 9, the current GitHub branch, existing notebooks, README files, comments, and previous AI-generated changes.
- Commit 9 is only a recovery scaffold and a source of historical implementation details.
- Existing code may be reused only when it is scientifically consistent with the paper.
- No reported paper result may be copied into generated output as though it were computed.
- No synthetic, mock, approximated, or reconstructed graph may silently replace an original processed graph.
- Every generated table and figure must be derived from saved run artifacts.
- Values absent from the paper must be recorded as reproduction assumptions, not described as paper-specified values.

## 3. Authoritative Dataset Artifacts

Use the four processed graph archives introduced in the repository's first commit and still present at commit 9:

- `elliptic_standardized.zip`
- `ieee_standardized.zip`
- `dgraphfin_reduced.zip`
- `amlsim_final_unified.zip`

The notebook must verify each archive using its SHA-256 digest before loading:

| Archive | SHA-256 |
|---|---|
| `elliptic_standardized.zip` | `0a86c749a1c388be50ca2a828485532216dd1cf29f39e9a5a69159c66514ee29` |
| `ieee_standardized.zip` | `e73460dffa01a807cdf113a768c535f8d21d9bf6a94e041ca863449d220bf761` |
| `dgraphfin_reduced.zip` | `c7726f1dfd085e548ac86ad1cae62d751c87df95248260499ab431b9bfbb9e21` |
| `amlsim_final_unified.zip` | `99dd1eb32ad2a71fd6eb95eaa942951599f88418c1fafe884957a143491acb1e` |

The loader must safely normalize both serialized formats found in the archives:

- A direct `HeteroData` object.
- A single-element `list[HeteroData]`.

It must not change labels, edges, node identities, or graph topology.

### Required Table 1 assertions

| Dataset | Transaction nodes | Entity nodes | Feature size | Total edges |
|---|---:|---:|---:|---:|
| Elliptic | 46,564 | 93,128 | 128 | 222,880 |
| IEEE-CIS | 590,540 | 14,962 | 128 | 2,952,700 |
| DGraphFin | 250,000 | 324,822 | 128 | 1,036,164 |
| AMLSim | 45 | 35 | 128 | 443 |

Additional assertions must verify:

- Elliptic contains 42,019 legitimate and 4,545 illicit transaction labels.
- Labels are binary with legitimate `0` and fraud/illicit `1`.
- Transaction feature width is 128.
- Account and merchant feature width is normalized to 64 without altering existing values.
- Required forward and reverse heterogeneous relations are present.
- All features and labels are finite.
- Every edge index is within the valid source and destination node ranges.

## 4. Paper Specification Matrix

Before implementing either notebook, create a machine-readable specification dictionary and a displayed human-readable table covering:

- Dataset names and roles.
- Node and edge types.
- Transaction and entity feature widths.
- Hidden dimension.
- Number of heterogeneous GraphSAGE layers.
- Type-specific projection order.
- Fraud-classification head.
- GRL and domain-classification behavior.
- Focal-loss equation.
- Optimizer, learning rate, and epochs.
- Random seeds.
- LODO source and target combinations.
- OOD source and target domains.
- Adaptive-OOD interpretation.
- Threshold-selection rule.
- Metrics.
- Required tables, figures, and ablations.

The notebook must print this specification before training so that implementation choices can be checked against the paper.

## 5. Ambiguity Register

The paper does not provide every value required for executable code and contains tension in its description of Adaptive OOD. Create an explicit ambiguity register for at least:

- Focal-loss `alpha`.
- Focal-loss `gamma`.
- Domain-loss weight `lambda`.
- GRL strength or schedule.
- Dropout probability.
- Validation fraction.
- Weight decay, if any.
- Exact intermediate classifier widths.
- Whether standard deviation is population or sample standard deviation.
- Adaptive-OOD use of target features versus target labels.

For each ambiguity, record:

1. The relevant paper statement.
2. What is missing or contradictory.
3. The selected implementation assumption.
4. Why that assumption is conservative.
5. Whether a sensitivity check is required.

Commit 9 values may be cited as historical evidence, but they must not be described as values specified by the paper.

## 6. Correct Scientific Protocol Shared by Both Notebooks

### 6.1 Reproducibility

- Use seeds `[42, 123, 3407, 2025, 9999]`.
- Train for 40 epochs in full-paper mode.
- Use deterministic PyTorch and CUDA settings where supported.
- Record Python, PyTorch, PyTorch Geometric, CUDA, GPU, NumPy, pandas, scikit-learn, and XGBoost versions.
- Record the complete configuration and dataset hashes in a manifest.
- Provide `SMOKE_TEST` and `FULL_PAPER_RUN` modes.
- Never present smoke-test results as paper-valid results.

### 6.2 Leakage-safe source validation

- Create deterministic, stratified transaction-node training and validation masks within each source domain.
- Fraud-label loss may use only source training labels.
- Threshold selection may use only pooled source validation labels and probabilities.
- Target labels must never influence training, early stopping, calibration, model selection, or threshold selection.
- Assert that all train, validation, adaptation, and evaluation index sets obey the intended separation.

### 6.3 DG-Hetero-GNN architecture

Implement the paper's components:

1. Type-specific feature projection:
   `Linear -> LayerNorm -> ReLU -> Dropout`.
2. Three relation-specific heterogeneous GraphSAGE layers.
3. Nonlinear activation following each message-passing layer.
4. A deep transaction fraud-classification head.
5. Gradient Reversal Layer.
6. Domain classifier.
7. Focal fraud-classification loss.
8. Validation-based adaptive threshold selection.

All relation directions supplied by the processed graphs must be preserved. Merchant and account information must have a valid message path into transaction embeddings.

### 6.4 Domain-generalization training

For the proposed DG-Hetero-GNN in LODO and standard OOD:

- Train fraud classification on source training labels.
- Train the domain classifier using source-domain identities.
- Apply GRL so the encoder learns source-domain-invariant representations.
- Use `L_total = L_fraud + lambda * L_domain` as specified in the paper.

The proposed model must not silently disable its domain-generalization branch.

### 6.5 LODO experiments

Implement exactly:

- LODO0: train on IEEE-CIS and DGraphFin; evaluate on Elliptic.
- LODO1: train on Elliptic and DGraphFin; evaluate on IEEE-CIS.
- LODO2: train on Elliptic and IEEE-CIS; evaluate on DGraphFin.

Repeat every proposed-model and baseline experiment across all five seeds.

### 6.6 Standard OOD

- Train on Elliptic, IEEE-CIS, and DGraphFin.
- Evaluate on AMLSim.
- Do not use AMLSim features or labels during standard OOD training or threshold selection.

### 6.7 Adaptive OOD

Use the interpretation that reconciles the paper's two statements:

- Target-domain AMLSim features may participate in domain-adversarial adaptation.
- AMLSim fraud labels must not participate in training or threshold selection.
- Source labels continue to drive focal fraud-classification loss.
- Domain loss uses source-domain identities and an AMLSim target-domain identity.
- AMLSim labels are revealed only for final evaluation.

Clearly label this as transductive, unlabeled-target adaptation. Also provide an audit assertion proving that target `y` is not accessed by the training function.

## 7. Baselines

Implement and evaluate:

- XGBoost.
- MLP.
- GCN.
- GraphSAGE.
- GAT.
- HeteroGraphSAGE.

For comparability:

- Use the same source training and validation indices for all applicable models.
- Use the same five seeds.
- Use the paper's shared hidden dimension and three-layer protocol where applicable.
- Select classification thresholds from source validation data rather than target labels.
- Keep target evaluation identical across models.
- Document architecture-specific deviations that cannot be shared with XGBoost or MLP.

Homogeneous graph baselines must use a documented conversion based on the original processed relations. The conversion must not invent edges or use target-only information.

## 8. Ablation Study

Re-train and evaluate the following variants across all five seeds:

- Full DG-Hetero-GNN.
- Without type-specific feature projection.
- Without heterogeneous message passing.
- Without domain generalization.
- Without the deep fraud classifier.
- Without threshold optimization, using fixed threshold 0.5.

The fixed-threshold ablation must be calculated for every seed. It must not report a first-seed value as a five-seed aggregate.

## 9. Metrics

Compute and retain per seed:

- ROC-AUC.
- PR-AUC.
- Precision.
- Recall.
- F1-score.
- Selected threshold.
- Expected Calibration Error.
- Brier score.
- True positives.
- True negatives.
- False positives.
- False negatives.

Report mean and standard deviation across the five seeds. Record the standard-deviation convention in the ambiguity register.

## 10. Figures, Tables, and Explanations

Generate the paper's required outputs:

- Table 1: graph statistics.
- Table 2: LODO results for all models.
- Table 3: detailed DG-Hetero-GNN LODO analysis.
- Table 4: OOD AMLSim results.
- Table 5: OOD versus Adaptive OOD.
- Figure 2(a): OOD precision-recall curves.
- Figure 2(b): OOD ROC curves.
- Figure 2(c): OOD baseline confusion heatmaps.
- Figure 2(d): OOD versus Adaptive-OOD heatmaps.
- Figure 2(e): calibration metrics across datasets.
- Figure 3(a): LODO ROC-AUC comparison.
- Figure 3(b): LODO PR-AUC comparison.
- Figure 3(c): LODO F1 comparison.
- Figure 3(d): ablation study.

Follow the presentation conventions in `Template.ipynb` without allowing that template to override the paper's scientific content:

- Use formal, journal-style titles and axis labels.
- Use a stable, explicit color palette.
- Display DG-Hetero-GNN consistently as the proposed model.
- Export publication-quality figures at 300 DPI.
- Print the numerical data underlying every visualization immediately below it.
- Save full curve coordinates, not only summary AUC values.
- Add a concise, data-grounded explanation below every table and figure.
- Explanations must describe computed results, not what a model "should" achieve.

## 11. Artifact Contract

Both notebooks must write the same artifact schema:

```text
artifacts/
  manifest.json
  graph_statistics.json
  results/
    lodo_*.json
    ood_*.json
    adaptive_ood_*.json
    ablation_*.json
  predictions/
    <scenario>_<model>_seed<seed>.npz
  models/
    <scenario>_<model>_seed<seed>.*
  figures/
    *.png
  tables/
    *.csv
```

Every result file must contain:

- Scenario name.
- Model name.
- Source and target domains.
- Seed.
- Configuration.
- Threshold provenance.
- Per-seed metrics.
- Aggregate metrics.
- Prediction artifact path.
- Dataset hashes.

Writes should be atomic where practical so interrupted runs do not leave valid-looking partial artifacts.

## 12. Phase 1: Minimal-Diff Notebook

Create `DG_Hetero_GNN_Paper_Minimal_Diff.ipynb` by preserving commit 9's structure wherever possible.

Required minimal corrections include:

- Replace hard-coded Google Drive paths with notebook-relative configurable paths.
- Add safe archive extraction and SHA-256 verification.
- Normalize list versus direct graph serialization.
- Preserve and validate the original graph topology.
- Normalize entity feature widths to 64.
- Correctly assign domain identities.
- Remove reliance on global device variables inside model forward methods.
- Introduce source training and validation masks.
- Remove target-label threshold optimization.
- Use the paper's Adam optimizer.
- Apply focal loss according to the paper's equation and documented assumptions.
- Enable domain-generalization training for the proposed model in LODO and OOD.
- Implement unlabeled-target Adaptive OOD.
- Connect and run all baseline models.
- Add the full ablation suite.
- Save complete per-seed artifacts.
- Add paper tables, figures, and explanations.

Do not refactor code merely for style in this notebook. Each nontrivial edit must be mapped to a paper requirement or an objective execution/correctness defect.

## 13. Phase 2: Clean Reproduction Notebook

Copy the validated minimal-diff notebook to `DG_Hetero_GNN_Paper_Clean_Reproduction.ipynb` and refactor without changing the scientific protocol.

Refactoring goals:

- Central configuration dataclass or validated dictionary.
- Dedicated data-loading and invariant-checking section.
- Reusable model factories.
- Reusable training and prediction functions.
- Explicit source, target, adaptation, and evaluation roles.
- One scenario runner for neural models.
- A separate, compatible runner for XGBoost.
- Central metric and threshold modules.
- Central artifact writer and loader.
- Central figure/table renderer.
- Reduced duplicated code.
- Clear Markdown explaining the connection between paper equations and implementation cells.

The clean notebook must not introduce new scientific behavior. Cross-notebook equivalence tests must compare predictions, thresholds, and metrics for at least smoke-test mode and one fixed full-data seed.

## 14. Validation Plan

### 14.1 Static validation

- Valid nbformat 4 notebook.
- Every code cell compiles.
- No stored error outputs.
- No missing cell IDs.
- Imports are available or installed through a documented environment cell.
- Paths are platform-independent.
- No hidden dependency on the current working directory.

### 14.2 Unit and invariant tests

- Archive hashes match.
- Table 1 assertions pass.
- Elliptic label counts pass.
- GRL reverses gradients with the expected sign.
- Model output dimensions are correct.
- Merchant and account perturbations can affect transaction logits through valid graph relations.
- Focal loss is finite and matches a hand-calculated small example.
- Train and validation masks are disjoint.
- Standard OOD has no target access.
- Adaptive OOD training cannot access target fraud labels.
- Threshold selection receives source validation arrays only.
- Confusion-matrix totals equal target sample counts.

### 14.3 Smoke validation

- Run AMLSim and small deterministic source subsets for one seed and a small number of epochs.
- Execute every model family.
- Execute every ablation code path.
- Render every table and figure using smoke artifacts.
- Mark all smoke outputs as non-paper-valid.

### 14.4 Full validation

- Run every LODO, OOD, Adaptive-OOD, baseline, and ablation experiment for five seeds and 40 epochs.
- Execute both notebooks from a clean kernel in order.
- Confirm no target-label leakage through access logging or guarded target-label objects.
- Confirm cross-notebook numerical equivalence within deterministic floating-point tolerance.
- Confirm every saved figure has a corresponding numerical table.
- Confirm all paper tables and figures are generated.

## 15. Comparison With Reported Paper Results

Create comparison tables containing:

- Paper-reported mean.
- Paper-reported standard deviation.
- Reproduced mean.
- Reproduced standard deviation.
- Absolute difference.
- Relative difference where meaningful.
- Pass/fail against a declared tolerance.

The tolerance must be declared before inspecting reproduced values. Suggested diagnostic tolerances are:

- Exact match for graph counts and deterministic invariants.
- `1e-6` for cross-notebook equivalence on the same stored predictions.
- A separately justified tolerance for stochastic comparison with the paper's rounded metrics.

A failed paper comparison must remain visible. The notebook must diagnose the discrepancy rather than replacing computed values with paper values.

## 16. Completion Criteria

The work is complete only when:

- Both notebooks exist at the requested location.
- Both use the paper as their sole normative specification.
- Both use the original processed graph artifacts and reproduce Table 1 exactly.
- Both execute from a clean kernel without errors.
- Both implement leakage-safe LODO, OOD, and Adaptive-OOD.
- Both execute all proposed-model, baseline, and ablation experiments.
- Both generate Tables 1-5 and Figures 2-3 from computed artifacts.
- Both print the numerical source data and a grounded explanation for every visualization.
- Both save complete five-seed artifacts and manifests.
- Both pass cross-notebook equivalence checks.
- Any remaining mismatch with the paper is explicitly reported and traced to a documented ambiguity or reproducibility limitation.

## 17. Non-Goals

- Do not repair or rewrite the GitHub repository history.
- Do not overwrite commit 9.
- Do not use the current broken notebooks as an authority.
- Do not reconstruct the graphs from unrelated raw-data assumptions.
- Do not tune against target evaluation labels.
- Do not fabricate missing experimental results.
- Do not claim exact reproduction when a paper ambiguity prevents it.

## 18. Failed-Run Prevention Addendum (Mandatory)

The stopped 2026-08-30 run is evidence that static notebook inspection is not sufficient. The following are release-blocking requirements for both deliverables:

1. Threshold selection must be an exact `O(n log n)` sorted/cumulative implementation of the source-validation F1 optimum, preserving the decision rule `probability >= threshold` and the original lowest-threshold tie break. It must be regression-tested against exhaustive search on randomized arrays and tied probabilities.
2. The archived heterogeneous relation set is authoritative. Do not synthesize reverse relations and do not symmetrize the homogeneous baseline edge list when the archive already stores semantic directions. Table 1 must count every archived edge exactly once and assert the exact archived relation schema.
3. Moving a graph to CUDA must never mutate the retained CPU graph. GPU feature views must be newly constructed; XGBoost and all NumPy consumers must receive CPU tensors. GPU views and models must be released after every seed.
4. Domain labels must be scenario-local contiguous IDs. The domain head width must equal the number of training domains: source domains only for ordinary DG, and source plus unlabeled target for adaptive OOD.
5. `without_domain_generalization` must disable both gradient reversal and domain-classification loss. HeteroGraphSAGE must have an independent classifier/trainer with no domain head, GRL, or domain loss.
6. Adaptive-OOD training may receive only a feature-only target view that has no `y` attribute. Target labels may be read only by a separate post-training evaluator. A guard test must fail if the training path can access target labels.
7. Artifacts must be isolated under an immutable run scope containing implementation ID, run mode, and run ID. A FULL manifest may reference only FULL artifacts from that same run. Existing mixed `artifacts/` output is quarantined legacy evidence and must never be reused.
8. Every scenario/model/seed writes an atomic checkpoint immediately on completion. A rerun with the same run ID resumes completed seeds only after validating its configuration and dataset hashes.
9. Progress must be unbuffered and flushed: run/scenario/model/seed start and finish, epoch heartbeat with elapsed time/loss, and CUDA allocated/reserved memory. The manifest status must transition through `running`, `complete`, or `failed` with timestamps.
10. Figures 2(a) and 2(b) must plot all OOD models in one figure rather than overwrite per model. Figure 2(e) must compare calibration across evaluation datasets, not models on only AMLSim. Every figure must have a saved numerical CSV and a generated explanation grounded in that CSV.
11. The manifest must record Python, PyTorch, PyG, CUDA/GPU, NumPy, pandas, scikit-learn, XGBoost, and Matplotlib versions. Dependencies must be pinned for the validated environment.
12. The executable notebook cells must contain the reviewed implementation itself, not silently delegate to a different unshown module. The minimal-diff and clean notebooks must be independently executable and cross-checked for identical predictions at `1e-6` on a deterministic smoke run.
13. No FULL_PAPER_RUN may start until archive/schema tests, threshold equivalence tests, CPU/GPU mutation tests, target-label guard tests, baseline independence tests, smoke execution, artifact-scope tests, and notebook source-equivalence tests all pass.
14. Focal-loss alpha/gamma and GRL/domain-loss assumptions require declared sensitivity runs. Paper comparison tables and tolerances must be created before interpreting reproduced results; mismatches remain visible.
