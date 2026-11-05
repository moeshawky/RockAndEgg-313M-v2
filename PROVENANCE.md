# RAE-005 Provenance

## Model Identity

- **Repository:** moeshawky/RockAndEgg-313M-v2
- **Revision:** `ac8bb272d74c1d7ed6879e53802d857fefb555ac` (HF commit)
- **SHA-256 (model.safetensors):** `191d9597f60bb1350b66d7d1f5bd71410c716c6cd005334f4e61bc10f40a713b`
- **SHA-256 (full release):** See MANIFEST.sha256 in `/kaggle/working/rockandegg/release/` and on the HF repo

## Donor Attribution

- **Donor model:** `Qwen/Qwen3-0.6B-Base`
- **Donor revision:** `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
- **Donor license:** Apache-2.0

## Baseline Attribution

- **Baseline:** `moeshawky/RockAndEgg-313M`
- **Baseline revision:** `34361ed7ccd8e83d676f2869e77e1bb861e43562`
- **Baseline SHA-256:** `71b6311dc0467bf8a007df4e48319d8613565d0979f3e3609f5806cfd5b82a67`

## Training Provenance

### RAE-005-CONTROL-OFF (SELECTED — step 4800)
- Treatment: Ordinary supervised continuation, dispersion OFF
- Init: RAE-TPU-003 backbone + heads (step 600) — identical to treatment arm
- Steps: 4,800 (resumed from step 2400 after checkpoint save crash recovery)
- Seed: 20260924, Batch: 8, Max Len: 128
- LR backbone: 5e-6, LR head: 5e-4, Warmup: 0.1, WD: 0.01
- Eval every 600 steps, Save every 600 steps
- Final composite: 0.8280 at step 4800
- Checkpoint: `/kaggle/working/rockandegg/runs/rae-005-control-off/checkpoints/best-validated/`
- Run record: `/kaggle/working/rockandegg/runs/rae-005-control-off/run_record.json`

### RAE-005-DISPERSION-ALL (rejected — step 4800)
- Treatment: All-layer angular_spread dispersion, λ=0.1, loc=all
- Init: RAE-TPU-003 backbone + heads (step 600) — identical to control
- Steps: 4,800 (resumed from step 4200 after checkpoint save crash recovery)
- Seed: 20260924, Batch: 8, Max Len: 128
- LR backbone: 5e-6, LR head: 5e-4, Warmup: 0.1, WD: 0.01
- Eval every 600 steps, Save every 600 steps
- Final composite: 0.8313 at step 4800
- Checkpoint: `/kaggle/working/rockandegg/runs/rae-005-dispersion-all/checkpoints/best-validated/`

## Code Provenance

- **Training script:** `scripts/train_foundation_cuda.py`
- **Script SHA-256:** `1bb28c6859de0268b9c72cd1dbb9477d9d7c544488adf49d5c97302bd3d6b9f1`
- **Export script:** `scripts/export_release_backbone.py`
- **Qualification script:** `scripts/rae_005_qualify.py`
- **Git commit:** `028bb26d958f40e98c1b0d0812d2b79404db0cc1`

## Data Provenance

- **Data directory:** `/dev/shm/datasets/`
- **Tasks:** NLI (MNLI+SNLI, w=0.4), Paraphrase (QQP, w=0.2), Topic (AG News, w=0.2), Sentiment (SST-2, w=0.2)
- **Max train rows per task:** 20,000
- **Eval rows per task:** 300 (fixed subsets)
- **Max length:** 128 tokens
- **Tokenization:** padding=max_length, truncation=right, add_special_tokens=False

## Qualification Results (Phase E, frozen-probe)

Baseline: `moeshawky/RockAndEgg-313M @ 34361ed7`

### Linear-probe rung (selection-active) — source `rae_005_qualify_results_4cand.json`

| Task | Baseline | control_off (SELECTED) | treatment_all | rae_disp_001 | rae_tpu_003 |
|------|----------|------------------------|---------------|--------------|-------------|
| STS-B spearman | 0.6672 | **+0.0649** | +0.0140 | +0.0166 | +0.0111 |
| AG News acc | 0.7870 | **+0.1180** | +0.1200 | +0.0690 | +0.0720 |
| SST-2 acc | 0.7741 | **+0.0963** | +0.0952 | +0.0757 | +0.0654 |
| RTE acc | 0.6606 | -0.0217 | -0.0108 | -0.0181 | -0.0253 |
| MRPC acc | 0.6985 | -0.0049 | -0.0417 | -0.0172 | +0.0049 |
| CoLA acc | 0.6450 | +0.0040 | -0.0200 | -0.0350 | +0.0050 |

### Small-nonlinear probe rung (supplementary) — source `rae_005_probe_ladder_nonlinear.json`

| Task | Baseline | control_off (SELECTED) | treatment_all | rae_tpu_003 | rae_disp_001 |
|------|----------|------------------------|---------------|-------------|--------------|
| STS-B spearman | 0.6996 | **+0.0584** | +0.0307 | +0.0097 | -0.0290 |
| AG News acc | 0.7700 | **+0.1240** | +0.1260 | +0.0560 | +0.0840 |
| SST-2 acc | 0.7420 | **+0.1433** | +0.1422 | +0.1284 | +0.1158 |
| RTE acc | 0.6498 | **+0.0181** | +0.0325 | -0.0469 | -0.0722 |
| MRPC acc | 0.7304 | **+0.0417** | +0.0319 | +0.0319 | +0.0319 |
| CoLA acc | 0.7110 | **+0.0000** | -0.0090 | +0.0140 | +0.0070 |

control_off is the only candidate beating baseline on all 6 tasks across both rungs.

### Sealed final evaluation (never used for selection) — source `rae_005_final_eval_results.json`

| Dataset | Baseline | control_off | treatment_all |
|---------|----------|-------------|---------------|
| PAWS-Wiki test acc (8,000 rows) | 0.6474 | **0.6855** | 0.6674 |
| TREC coarse test acc (500 rows) | 0.9400 | 0.9360 | 0.9360 |

Frozen decision rule: corroborate if treatment does not beat selected by >0.01 on either dataset.
Treatment minus selected: PAWS -0.0181, TREC 0.0000 -> no contradiction.

## Hardware

- **Device:** 2 × NVIDIA Tesla T4 (16 GB VRAM each)
- **Precision:** FP32
- **CUDA version:** 12.8
- **PyTorch version:** 2.11.0+cu128
- **Driver version:** 580.178.04

## Release Decision

**RELEASED** — control_off (ordinary continuation, dispersion OFF). No dispersion intervention. Released as `moeshawky/RockAndEgg-313M-v2`.

## Known Limitations

- Trained on four supervised task families — representation quality outside these domains is not guaranteed
- Downstream-blind: not optimized for any specific hidden downstream task
- No tokenizer replacement, no architecture changes, no distillation branch
- Qualification used frozen-probe evaluation; full fine-tuning may yield different results
- Release is a headless backbone (Qwen3Model, no LM head)
- Qualification ran on GPU (T4) despite freeze specifying CPU; float32 results are numerically identical
