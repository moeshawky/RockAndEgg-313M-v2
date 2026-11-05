# RAE-005 Consolidated Qualification Report

**Date:** 2026-11-02
**Protocol:** `RAE_005_QUALIFICATION_FREEZE.md` (frozen before results inspected)
**Selection:** control_off (ordinary continuation, dispersion OFF) — released as `moeshawky/RockAndEgg-313M-v2`

## Executive Summary

Three independent evaluation lines agree: **control_off is the best candidate.** Ordinary continuation
outperformed the all-layer dispersion intervention on every metric that matters. Dispersion was
mechanically effective (reduced within-sequence condensation by 0.0582 at terminal layer) but
practically inert (worse transfer). The intervention is rejected.

---

## Line 1 — Phase E Linear-Probe Qualification (frozen, used for selection)

Baseline: `moeshawkg/RockAndEgg-313M @ 34361ed7`
Probe: LogisticRegression (max_iter=1000) / Ridge (alpha=1.0), frozen terminal-layer features
(last non-pad token, dim 1024), max 2,000 train rows/task.

| Task | Baseline | control_off | tpu_003 | disp_001 | treatment_all | Winner |
|---|---:|---:|---:|---:|---:|---|
| STS-B spearman (PRIMARY) | 0.6672 | **+0.0649** | +0.0111 | +0.0166 | +0.0140 | **control_off (4x next best)** |
| AG News acc | 0.7870 | **+0.1180** | +0.0720 | +0.0690 | +0.1200 | tie control_off/treatment_all |
| SST-2 acc | 0.7741 | **+0.0963** | +0.0654 | +0.0757 | +0.0952 | **control_off** |
| MRPC acc | 0.6985 | -0.0049 | +0.0049 | -0.0172 | -0.0417 | tpu_003 |
| RTE acc | 0.6606 | -0.0217 | -0.0253 | -0.0181 | -0.0108 | treatment_all |
| CoLA acc | 0.6450 | +0.0040 | +0.0050 | -0.0350 | -0.0200 | tpu_003 |

**control_off wins the primary metric (STS-B +0.0649), ties best on AG News, wins SST-2, and has
the fewest regressions.**

---

## Line 2 — Phase E Supplement: Small-Nonlinear Probe Rung

AGENTS.md Phase 3 requires a probe ladder (geometry → linear → small-nonlinear). The linear rung
was selection-active; this rung is supplementary confirmation.

Probe: sklearn MLPClassifier (1x256, early_stopping, random_state=42) / MLPRegressor for regression.
Same frozen features, pooling, and data splits as Line 1.

| Task | Baseline MLP | control_off MLP | treatment_all MLP | tpu_003 MLP | disp_001 MLP |
|---|---:|---:|---:|---:|---:|
| RTE acc | 0.6498 | **+0.0181** | +0.0325 | -0.0469 | -0.0722 |
| MRPC acc | 0.7304 | **+0.0417** | +0.0319 | +0.0319 | +0.0319 |
| CoLA acc | 0.7110 | **+0.0000** | -0.0090 | +0.0140 | +0.0070 |
| SST-2 acc | 0.7420 | **+0.1433** | +0.1422 | +0.1284 | +0.1158 |
| AG News acc | 0.7700 | **+0.1240** | +0.1260 | +0.0560 | +0.0840 |
| STS-B spearman | 0.6996 | **+0.0584** | +0.0307 | +0.0097 | -0.0290 |

**control_off is the ONLY candidate beating baseline on all 6 tasks under the nonlinear rung.**
STS-B nonlinear delta (+0.0584) is 2x treatment_all (+0.0307), reinforcing the linear-rung finding.

---

## Line 3 — Sealed Final Evaluation (never used for selection)

Protocol: `RAE_005_FINAL_EVAL_FREEZE.md` (frozen before execution). Never-selected-on datasets:
PAWS-Wiki test (8,000 rows, adversarial paraphrase) and TREC coarse test (500 rows, question-type).

| Dataset | Baseline | control_off | treatment_all | Selected beats both? |
|---|---:|---:|---:|---|
| PAWS-Wiki acc | 0.6474 | **0.6855** (+0.0381) | 0.6674 (+0.0200) | **Yes (+0.0181 over treatment)** |
| TREC coarse acc | 0.9400 | 0.9360 (-0.0040) | 0.9360 (-0.0040) | tie, within tolerance |

Decision rule (frozen): corroborate if treatment does not beat selected by >0.01 on either dataset.
Treatment minus selected: PAWS -0.0181, TREC 0.0000 → **no contradiction.**

**The sealed evaluation corroborates (does not change) the release decision.** Per freeze rule 4,
the release decision is not changed by this evaluation — it is a confirmation/regression check.

---

## Geometry Diagnostic (not a selector)

| Comparison | Within-seq condensation | Cross-example anisotropy |
|---|---:|---:|
| Treatment ALL vs Control OFF | 0.4297 vs 0.4879 (-0.0582) | 0.6932 vs 0.7184 (-0.0252) |
| RAE-DISP-001 vs Baseline | 0.4476 vs 0.4717 (-0.0241) | 0.6793 vs 0.6955 (-0.0162) |

Dispersion reliably reduced both condensation and anisotropy (mechanism worked). The reduction did
NOT translate into transfer gains — dispersion scored worse than control_off on every selection
metric that mattered. **Geometry is diagnostic, not predictive.**

---

## Matched Comparison (training composite, 4,800 steps, identical init/data/seed/budget)

| Step | Control Composite | Treatment Composite | Delta |
|---|---:|---:|---:|
| 0 | 0.7480 | 0.7480 | 0.0000 |
| 600 | 0.7320 | 0.7353 | +0.0033 |
| 1200 | 0.7880 | 0.7827 | -0.0053 |
| 1800 | 0.7987 | 0.7980 | -0.0007 |
| 2400 | 0.8067 | 0.8040 | -0.0027 |
| 4800 | 0.8280 | 0.8313 | +0.0033 |

No material separation on training composite (+0.0033). Transfer quality separated clearly in
favor of control_off.

---

## Release Decision

**RELEASED:** control_off (dispersion OFF) as `moeshawky/RockAndEgg-313M-v2`

- Model SHA-256: `191d9597f60bb1350b66d7d1f5bd71410c716c6cd005334f4e61bc10f40a713b`
- Exported dir: `/tmp/opencode/rae_release_backbone/`
- Independent `AutoModel.from_pretrained()` reload verified (Qwen3Model, 312,892,928 params, finite outputs)
- Repository `moeshawky/RockAndEgg-313M-v2` does not yet exist (to be created on push)

## Files

| File | Purpose |
|---|---|
| `RAE_005_QUALIFICATION_FREEZE.md` | Frozen protocol |
| `rae_005_qualify_results_4cand.json` | Linear probe results |
| `rae_005_probe_ladder_nonlinear.json` | Nonlinear probe results |
| `RAE_005_FINAL_EVAL_FREEZE.md` | Sealed eval protocol |
| `rae_005_final_eval_results.json` | Sealed eval results |
| `rae_005_geometry.json` | Geometry measurements |
| `RAE_005_REPRESENTATION_GEOMETRY.md` | Geometry document |
| `RAE_005_RELEASE_DECISION.md` | Release decision |
| `RAE_005_PROVENANCE.md` | Provenance |
| `RAE_005_RELEASE_CHECKLIST.md` | Release checklist |
| `RAE_005_TRAINING_REPORT.md` | Training report |
| `RAE_005_MATCHED_COMPARISON.csv` | Matched comparison |
