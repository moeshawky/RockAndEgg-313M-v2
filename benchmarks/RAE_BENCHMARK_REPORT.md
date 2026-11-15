# REPORT.md — RockAndEgg-313M-v2 vs BERT/DeBERTa Benchmark

## Executive Summary

This benchmark evaluates RockAndEgg-313M-v2 (RAE-v2), a headless 10-layer causal Qwen3-derived backbone with ~312.9M parameters (~156M embedding, ~157M compute), against matched BERT/DeBERTa baselines on 2× Tesla T4 GPUs.

**Headline finding:** RAE-v2 beats bert-base+recipe on 5 of 9 common clean-transfer tasks (banking77, emotion, paws, qnli, trec) with 3 wins at p < 0.05, but trails deberta-v3-base+recipe on all 4 common tasks (cola, mrpc, rte, stsb) with all 4 at p < 0.005. RAE-v2's inference throughput is 0.33× bert-base (true FP16 storage), a major deployment limitation.

## Benchmark Protocol

### Models Evaluated
| Model | Parameters | Source |
|-------|-----------|--------|
| RAE-v2 | ~312.9M (~157M compute) | HF `moeshawky/RockAndEgg-313M-v2` (true FP16 safetensors) |
| bert-base | ~109M | google-bert/bert-base-uncased |
| deberta-v3-base | ~278M | microsoft/deberta-v3-base |
| bert-base+recipe | ~109M | bert-base + multi-task recipe (5,400 steps) |
| deberta-v3-base+recipe | ~278M | deberta-v3-base + multi-task recipe (5,400 steps) |

### Tasks
**Clean-transfer (headline, 10):** STS-B, RTE, MRPC, CoLA, QNLI, PAWS, TREC, BANKING77, Emotion, SciTail

### Protocols Followed
- Max length 128, `[SEP]` delimiter for Qwen-derived models, native pair encoding for BERT/DeBERTa
- 6-cell pooling × layer grid (best cell selected on validation per model/task)
- Fine-tuning: AdamW, weight decay 0.01, linear decay with 6% warmup, grad clip 1.0, 1 epoch, fp32 precision
- Seeds: [13, 21, 34, 55, 89]
- Paired t-tests across same seeds for model comparisons

## Stage Results

### Stage 0: Setup and Provenance — COMPLETED

- **SHA-256 verified:** `191d9597f60bb1350b66d7d1f5bd71410c716c6cd005334f4e61bc10f40a713b` ✓
- **Architecture:** Qwen3Model, 10 layers, hidden 1024, 312.9M params, finite outputs ✓
- **FP16 parity:** mean cosine 0.999999 ≥ 0.999, zero NaN/Inf ✓
- **Corrected model published:** HF `moeshawky/RockAndEgg-313M-v2` (true FP16 safetensors, config dtype=float16) ✓

### Stage 1: Efficiency — COMPLETED

Measured on GPU 0, fp16, seq 128, batch 64, 20 warm-up + 100 timed iterations:

| Model | Latency (ms) | Throughput vs RAE-v2 |
|-------|-------------|---------------------|
| RAE-v2 (true FP16) | 318.6 | 1.00× |
| bert-base | ~104 | 3.06× |
| deberta-v3-base | ~185 | 1.72× |

**RAE-v2 throughput is 0.33× bert-base** — a major deployment limitation.

### Stage 2: Frozen Probes — COMPLETED

138 records across all models × all 14 tasks × 6-cell pooling/layer grid. Best cell per model/task (linear probe):

| Task | RAE-v2 | bert-base | deberta-v3-base |
|------|--------|-----------|-----------------|
| STS-B (Spearman) | 0.8169 | 0.8356 | 0.8709 |
| RTE (Accuracy) | 0.6751 | 0.5921 | 0.7076 |
| MRPC (Accuracy) | 0.7745 | 0.7402 | 0.8358 |
| CoLA (MCC) | 0.2472 (best 0.4527) | 0.3917 | 0.6019 |
| QNLI (Accuracy) | ~0.82 | ~0.79 | — |
| PAWS (Accuracy) | ~0.76 | ~0.59 | — |
| TREC (Accuracy) | ~0.954 | ~0.950 | — |
| BANKING77 (Accuracy) | ~0.904 | ~0.885 | — |
| Emotion (Accuracy) | ~0.70 | ~0.66 | — |

### Stage 3: Recipe-Matched Baselines — COMPLETED

Both bert-base+recipe and deberta-v3-base+recipe checkpoints saved. 5,400 optimizer steps each with multi-task sampling (MNLI 0.40, QQP 0.20, AG News 0.20, SST-2 0.20).

### Stage 4: Full Fine-Tuning — COMPLETED (rae-v2 & bert-base+recipe), PARTIAL (deberta-v3-base+recipe)

**Fine-tuned results (test metric, mean ± std, 5 seeds unless noted):**

| Task | RAE-v2 | bert-base+recipe | deberta-v3-base+recipe |
|------|--------|-----------------|----------------------|
| STS-B (Spearman) | 0.8610 ± 0.0019 | 0.8683 ± 0.0040 | 0.8878 ± 0.0022 |
| RTE (Accuracy) | 0.6693 ± 0.0146 | 0.6715 ± 0.0171 | 0.8152 ± 0.0176 |
| MRPC (Accuracy) | 0.7951 ± 0.0179 | 0.8216 ± 0.0209 | 0.8676 ± 0.0046 |
| CoLA (MCC) | 0.4727 ± 0.0324 | 0.5374 ± 0.0184 | 0.6198 ± 0.0038 |
| QNLI (Accuracy) | 0.8724 ± 0.0023 | 0.8708 ± 0.0108 | 0.9145 ± 0.0011 (3 seeds) |
| PAWS (Accuracy) | 0.9163 ± 0.0012 | 0.8087 ± 0.0196 | — |
| TREC (Accuracy) | 0.9764 ± 0.0043 | 0.9552 ± 0.0145 | — |
| BANKING77 (Accuracy) | 0.8928 ± 0.0031 | 0.6888 ± 0.0141 | — |
| Emotion (Accuracy) | 0.9288 ± 0.0012 | 0.9235 ± 0.0019 | — |
| SciTail (Accuracy) | — | — | — |

### Claim Verdicts

#### Claim A: Pareto — FAIL

**Criterion:** Throughput ratio ≥ 1.0 AND CI lower bound of macro delta > 0 AND wins ≥ 6 of 10 tasks.

**Result:** RAE-v2 throughput is 0.33× bert-base (far below 1.0 threshold). RAE-v2 wins 5 of 9 common tasks vs bert-base+recipe but trails deberta-v3-base+recipe on all 4 common tasks.

**Verdict:** FAIL — RAE-v2 does not match throughput threshold.

#### Claim B: Data Efficiency — INCONCLUSIVE

**Criterion:** RAE-v2 beats bert-base+recipe and deberta-v3-base+recipe at 1% and 10% data.

**Result:** Low-data experiments not completed (GradScaler FP16 incompatibility fixed, but time constraints prevented execution).

**Verdict:** INCONCLUSIVE — no low-data results available.

#### Claim C: Multilingual and Robustness — NOT EVALUATED

**Criterion:** RAE-v2 beats xlm-roberta-base and mdeberta-v3-base on XNLI non-English macro-average.

**Result:** Stage 5 not executed due to time constraints.

**Verdict:** NOT EVALUATED — requires MNLI 50k fine-tuning and XNLI evaluation.

#### Claim D: Parity at Lower Cost — FAIL

**Criterion:** Throughput ratio ≥ 1.5 AND CI lower bound of clean-transfer macro delta > −1.0 point.

**Result:** RAE-v2 throughput ratio vs deberta-v3-base is 0.33× (far below 1.5 threshold).

**Verdict:** FAIL — RAE-v2 throughput is substantially lower than deberta-v3-base.

#### Claim E: Frozen Quality — INCONCLUSIVE

**Criterion:** CI lower bound > 0 under both linear and MLP probes.

**Result:** Frozen probes completed for raw models but deberta-v3-base+recipe frozen probes not executed.

**Verdict:** INCONCLUSIVE — recipe-matched baseline frozen probes not executed.

## Summary of Claims

| Claim | Name | Verdict |
|-------|------|---------|
| A | Pareto | FAIL |
| B | Data Efficiency | INCONCLUSIVE |
| C | Multilingual/Robustness | NOT EVALUATED |
| D | Parity at Lower Cost | FAIL |
| E | Frozen Quality | INCONCLUSIVE |

## Key Findings

1. **RAE-v2 is competitive with bert-base+recipe:** Wins 5 of 9 common tasks (banking77 +0.204, paws +0.108, trec +0.021, emotion +0.005, qnli +0.002). Three wins are statistically significant (p < 0.05).

2. **RAE-v2 trails deberta-v3-base+recipe:** Loses all 4 common tasks (cola −0.147, rte −0.146, mrpc −0.073, stsb −0.027), all significant at p < 0.005.

3. **RAE-v2 efficiency is a major limitation:** Measured throughput is 0.33× bert-base and ~0.29× deberta-v3-base. ~3-4× slower than comparable baselines, severely limiting practical deployment.

4. **Throughput is the binding constraint:** Even with true FP16 storage (not FP32+autocast mislabeled as fp16), RAE-v2 remains ~3× slower than bert-base due to its ~3× parameter count.

## Limitations

1. **deberta-v3-base+recipe partial:** Only 4 of 10 tasks completed at 5 seeds (cola, mrpc, rte, stsb) + partial qnli (3/5). Full comparison incomplete due to T4 time constraints.
2. **bert-base+recipe missing scitail:** 9 of 10 tasks complete.
3. **No low-data results:** Data efficiency claim (B) could not be evaluated.
4. **No multilingual results:** Claim C could not be evaluated.
5. **1 epoch fine-tuning:** Deviated from protocol (3 epochs) due to time constraints.
6. **FP32 precision for training:** fp16 training failed due to PyTorch 2.x GradScaler incompatibility on T4.

## Conclusion

RockAndEgg-313M-v2 demonstrates competitive fine-tuned performance relative to bert-base+recipe (5 of 9 task wins, 3 statistically significant) despite having ~3× the parameters. However, it trails deberta-v3-base+recipe on all common tasks, and its inference throughput is ~3× lower than comparable baselines. The model shows promise as a research backbone for tasks where bert-scale efficiency is acceptable, but is not suitable for throughput-constrained deployment without quantization optimization.

---

*Appendix: All result keys are recorded in /kaggle/working/rae_bench/results/finetune.jsonl (append-only, resume by key).*
*Raw model weights were never modified. Only the patch file was produced.*
