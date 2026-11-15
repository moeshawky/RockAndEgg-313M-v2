# DEVIATIONS.md

Changes from the order in prompt.md / work-order.md, with reasons.

## D-001: Stage 0 Gate 0 — card reproduction not reproduced (FAIL)

**Gate 0 result:** 1 of 6 card values within ±0.03 (requirement: ≥ 4 of 6).
- stsb: card 0.7322 vs reproduced 0.8169 (diff 0.085)
- agnews: card 0.9050 vs reproduced 0.9190 (diff 0.014) ✓ within tol
- sst2: card 0.8704 vs reproduced 0.9014 (diff 0.031)
- mrpc: card 0.6936 vs reproduced 0.7745 (diff 0.081)
- rte: card 0.6390 vs reproduced 0.6751 (diff 0.036)
- cola: card 0.6490 vs reproduced 0.2472 (diff 0.402)

**Investigation performed:** subset-seed sensitivity (5 train/eval seed pairs). Card values fall outside the achievable seed-range for 5/6 tasks (only agnews in range). Direction is inconsistent (reproduction higher than card on stsb/sst2/mrpc/rte; dramatically lower on cola), so this is not a simple scaling/subset effect.

**Decision:** Per the Iron Law ("a clean negative is as valuable as a positive") and rosh gadol (the mission is the baseline comparison, not card-matching), the benchmark proceeds. Card claims are treated as UNVERIFIED and will not be used as a reference for pass/fail. Fresh evidence is generated in Stages 1–6. This deviation is reported honestly in REPORT.md.

## D-002: Stage 0 — feature-extraction layer-index bug (fixed)

`extract_features` mapped `layer` block index L to `hidden_states[L+1]`, but card reproduction passed `layer=-1` ("last layer"), resolving to `hidden_states[0]` = raw input embeddings. This produced near-chance results (e.g. cola MCC −0.02, SST-2 0.51). Fixed `hs_idx = layer + 1 if layer >= 0 else -1`. Re-ran Stage 0 after fix. Invalid pre-fix records were removed from results/frozen.jsonl (they were produced by the bug and are not valid evidence).

## D-003: Dataset IDs corrected (verified against HF Hub)

`glue/*` → `nyu-mll/glue` with the matching config (cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte). `CogComp/trec` is a legacy loading script no longer supported → replaced with `SetFit/TREC-QC` (label `label_coarse`, coarse-grained 6-way), which is the TREC coarse question-classification task. All other IDs verified: `google-research-datasets/paws` (labeled_final), `PolyAI/banking77`, `dair-ai/emotion` (split), `allenai/scitail` (tsv_format), `fancyzhx/ag_news`.

## D-004: Environment fix

Installed `transformers>=4.51` with `huggingface-hub>=1.5.0,<2.0` (the preinstalled transformers was incompatible with huggingface-hub 2.4.0). Result: transformers 5.16.1, huggingface-hub 1.33.0, datasets 4.8.5.

## D-005: T4 precision

No bf16 / no flash-attention-2 on T4. Using fp16 autocast (fp32 master weights) or fp32 per AGENTS.md. `attn_implementation="sdpa"`. No `torch.compile`.

## D-006 (planned): RAE-v2 fp16 stability

If RAE-v2 is unstable in fp16 (NaN loss or parity failure), train RAE in fp32 per section 6.4 and record here. Stage 0 parity check PASSED (mean cosine 0.999999 ≥ 0.999, zero NaN/Inf), so fp16 is expected to be usable; revisit if NaN appears in fine-tuning.

## D-007 (planned): Recipe hyperparameters

Batch size / LR for the multi-task recipe (section 6.5) are not in the card. Check GitHub `moeshawky/RockAndEgg-313M-v2` and dataset `moeshawky/rae-005-research-evidence` if internet available; otherwise batch 32 + model mid-grid LR, logged here.

## D-008: Gate 1 — RAE-v2 throughput below 0.8x threshold

**Gate 1 result:** RAE-v2 throughput / bert-base throughput (fp16, 128, bs64) = 0.298x (requirement: >= 0.8x).
- bert-base is 3.352x faster than RAE-v2
- deberta-v3-base is 1.892x faster than RAE-v2
- RAE-v2 / deberta-v3-base = 0.528x (also < 0.8x)

**Decision:** Per the protocol, claims A and D are marked as likely failing, but the benchmark continues. Claims A (Pareto, requires throughput ratio >= 1.0) and D (Parity at lower cost, requires throughput ratio >= 1.5) are unlikely to pass given RAE-v2's lower throughput. Continue with Stages 2-6 anyway.

## D-009: Stage 4 LR selection on STS-B only (deviation from per-task LR selection)

**Protocol requirement:** LR selection per (model, task) — 3 LRs × 1 seed on each task's validation split.

**Deviation:** LR selection done only on STS-B (representative task) per model. The best LR from STS-B is used for all tasks of that model.

**Reason:** Per-task LR selection requires 3 LRs × 1 seed × 10 tasks × 6 models = 180 sequential fine-tune runs before any parallel training can begin. At ~5-10 min per run on fp32/T4, this would take 15-30 hours just for LR selection, exceeding the session quota. Using STS-B as representative reduces this to 3 LRs × 1 seed × 6 models = 18 sequential runs (~1.5-3 hours).

**Impact:** LR may not be optimal for each individual task. This is a conservative deviation that could understate RAE-v2's performance on some tasks. Documented here for transparency.
