---
language:
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction
base_model: Qwen/Qwen3-0.6B-Base
tags:
  - qwen3
  - model-surgery
  - depth-pruning
  - representation-learning
  - text-classification
  - transfer-learning
  - causal-transformer
  - safetensors
---

# RockAndEgg-313M-v2

**A 313M-parameter, Qwen3-derived backbone for text classification, semantic representation learning, and downstream task adaptation.**

RockAndEgg-313M-v2 is the second foundation release of RockAndEgg, an independent model-development project investigating how far a compact, decoder-derived transformer can be adapted for discriminative language understanding.

Built from the first 10 transformer layers of [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), RockAndEgg retains the donor's hidden dimensions, attention topology, tokenizer, and vocabulary while removing 18 of its 28 transformer blocks.

Version 2 adds multi-task representation training and was selected through a controlled study of auxiliary dispersion regularization. Compared with the [original RockAndEgg-313M](https://huggingface.co/moeshawky/RockAndEgg-313M), it achieves stronger frozen-feature performance on several language-understanding benchmarks, notably semantic similarity, topic classification, and sentiment analysis.

The model is released without a language-modeling head or task-specific classification head. Downstream applications define their own labels, objectives, and output layers.

> **28 layers → 10 layers. 596M → 313M parameters. Same Qwen3 architecture family, with continued training for transferable representations.**

## Why "RockAndEgg"?

The name comes from an old Egyptian children's game.

Children strike decorated eggs against one another, and the egg that survives wins. There is, however, a particularly effective way to cheat: paint a stone to resemble an egg.

The question that gave the project its name was simple:

*What would an ancient Egyptian classifier be?*

A classifier worthy of the name should distinguish the painted stone from the real egg.

RockAndEgg is the foundation that learns the representations. A downstream classification head is what ultimately catches the cheat.

## Model details

| Property | Specification |
|---|---|
| Model | RockAndEgg-313M-v2 |
| Developer | moeshawky |
| Release | November 2026 |
| Architecture | Headless Qwen3Model |
| Parameters | 312,892,928 |
| Original model | Qwen3-0.6B-Base |
| Transformer layers | 10 of 28 (blocks 0–9) |
| Hidden size | 1,024 |
| Intermediate size | 3,072 |
| Attention heads | 16 |
| KV heads | 8 |
| Attention mechanism | Causal, grouped-query attention |
| Vocabulary | 151,936 tokens |
| Normalization | RMSNorm |
| Weight format | FP32 safetensors |
| Configured context | 32,768 positions |
| Evaluated sequence length | 128 tokens |
| Intended interface | Hugging Face `AutoModel` |
| License | Apache 2.0 |

### Model lineage

```text
Qwen3-0.6B-Base
    │
    │ Physical depth truncation
    │ Retain transformer blocks 0–9
    │ Preserve width, tokenizer and attention
    ▼
RockAndEgg-313M
    │
    │ Multi-task foundation training
    │ RAE-TPU-003
    ▼
Foundation checkpoint (step 600)
    │
    │ RAE-005 controlled continuation
    │ 4,800 steps
    │ Dispersion OFF selected
    ▼
RockAndEgg-313M-v2
    │
    ├── Classification heads
    ├── Regression heads
    ├── Ranking/scoring heads
    └── Other task-specific adaptations
```

### Why 10 layers?

The original architecture was selected through physical model-surgery experiments, not merely by modifying the configuration file.

The investigation evaluated multiple truncation depths against a full-depth Qwen3 control, including downstream fine-tuning, representation diagnostics, architecture ablations, and held-out testing.

In the original task-specific architecture study, the 10-layer model retained much of the 28-layer model's discriminative performance:

| Architecture | Parameters | Held-out macro-AUPRC |
|---|---:|---:|
| Full Qwen3, 28 layers | 596.1M | 0.9864 |
| Truncated Qwen3, 10 layers | 312.9M | 0.9850 |

These measurements were obtained from separately task-fine-tuned descendants on a specific eight-output semantic-classification application. They support the architectural choice; **they are not zero-shot results for the released headless foundation model**.

The 10-layer architecture represents a 64.3% reduction in transformer depth and approximately 47.5% fewer parameters.

Unlike a width-pruned or distilled replacement, the released backbone retains the original Qwen3 dimensional structure and uses the actual pretrained weights of the retained blocks.

The original development also investigated FFN narrowing and teacher-student distillation. Neither produced a better selected operating point under its tested conditions. The released RockAndEgg backbone is therefore not a MiniLM-style distilled student.

## Intended uses

RockAndEgg is designed as a reusable backbone for supervised language-understanding tasks.

Suitable applications include text classification, sentiment analysis, natural language inference, paraphrase detection, semantic similarity regression, and other tasks where contextual representations can be adapted using a trained output head.

It may also be useful for investigating depth-truncated causal transformers, representation geometry, and transfer learning under constrained model sizes.

**RockAndEgg is not a pretrained zero-shot classifier.** It has no fixed label ontology or trained universal decision head.

It is also not a chat model, instruction-following model, or competent general-purpose text generator. Although it retains causal attention, its released weights were selected for discriminative transfer rather than generation.

The use of a causal decoder backbone should not be confused with the bidirectional attention used by BERT or DeBERTa.

### Research objective

The broader RockAndEgg program aims to develop a compact backbone capable of outperforming appropriately matched BERT and DeBERTa baselines on the same downstream tasks.

**That objective has not yet been established experimentally.**

The benchmarks in this card compare RockAndEgg checkpoints against one another. They should not be interpreted as evidence that v2 outperforms BERT, DeBERTa, or other independently trained architectures.

## Training

### Multi-task foundation training

RockAndEgg-v2 continues from the RAE-TPU-003 foundation checkpoint, which adapted the original backbone through four supervised task families.

| Task family | Data source | Sampling weight |
|---|---|---:|
| Natural language inference | MNLI | 0.40 |
| Paraphrase classification | QQP | 0.20 |
| Topic classification | AG News | 0.20 |
| Sentiment classification | SST-2 | 0.20 |

The research source manifests also include SNLI. In the earlier foundation run, however, MNLI exhausted the configured NLI sample cap before SNLI examples were consumed. Effective SNLI contribution to the subsequent RAE-005 continuation is not established by the summarized release record and should not be assumed.

The training setup used up to 20,000 examples per supervised task, a maximum sequence length of 128 tokens, and temporary classification heads for the four task objectives.

These heads provided the training signal and were removed from the final backbone export.

The foundation phase completed 600 optimizer steps on TPU with a reported validation composite of 0.7453, compared with 0.347 at initialization.

### RAE-005 continuation

The subsequent experiment investigated whether continued supervised training could benefit from an additional representation-dispersion objective.

Two configurations were trained from the same foundation checkpoint:

| Configuration | Objective |
|---|---|
| `control_off` | Ordinary multi-task continuation |
| `treatment_all` | Same continuation plus all-layer angular dispersion, λ = 0.1 |

Both runs used 4,800 optimizer steps, FP32 precision, NVIDIA Tesla T4 hardware, and matched initialization, data schedule, optimizer settings, and seed.

The released v2 checkpoint is **`control_off`**, selected according to the experiment's frozen downstream evaluation criteria.

### Dispersion-loss research

The auxiliary objective was based on:

**Liu et al. (2026), [Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](https://proceedings.mlr.press/v306/liu26dy.html), ICML 2026.**

The paper identifies embedding condensation, a geometric phenomenon in which token representations become concentrated in a narrow directional region, and proposes dispersion loss as a regularization strategy.

Its experiments include Qwen3-0.6B, making the work particularly relevant to RockAndEgg's donor architecture.

RockAndEgg's experimental implementation adapted the authors' publicly released [LM-Dispersion](https://github.com/KrishnaswamyLab/LM-Dispersion) approach, using the `angular_spread` objective.

Earlier RAE-DISP experiments investigated terminal-layer dispersion under shorter training budgets. RAE-005 extended the comparison to all-layer regularization and a longer training horizon.

The release decision depended on downstream transfer measurements, not on whether the auxiliary objective successfully changed representation geometry.

## Evaluation

All results below concern frozen-backbone representation evaluation unless stated otherwise.

For the primary qualification, features were obtained from the last non-padding token of each sequence. A separate linear or nonlinear probe was then trained for the downstream task.

The principal evaluation used up to 2,000 probe-training examples and 1,000 evaluation examples per task, with 128-token truncation.

Sentence-pair tasks used the literal delimiter ` [SEP] ` between the two inputs.

The reported results are specific to this protocol. They are not full-model fine-tuning benchmarks.

### Linear-probe performance

Comparison against the original RockAndEgg-313M release:

| Benchmark | Metric | v1 | v2 | Δ |
|---|---|---:|---:|---:|
| STS-B | Spearman | 0.6672 | **0.7322** | **+0.0649** |
| AG News | Accuracy | 0.7870 | **0.9050** | **+0.1180** |
| SST-2 | Accuracy | 0.7741 | **0.8704** | **+0.0963** |
| MRPC | Accuracy | **0.6985** | 0.6936 | -0.0049 |
| RTE | Accuracy | **0.6606** | 0.6390 | -0.0217 |
| CoLA | Accuracy | 0.6450 | **0.6490** | +0.0040 |

STS-B was the principal independent-transfer metric used in release selection. Unlike AG News and SST-2, it is not a direct match to the four continuation-training output tasks.

The increase in STS-B Spearman provides evidence that some of the training benefits transfer beyond the immediate supervision objectives.

However, performance is not uniform. RTE and MRPC regress under linear probing.

AG News and SST-2 results should also be interpreted with particular care because those benchmark families were used during continuation training.

### Nonlinear-probe evaluation

A small multilayer perceptron with one 256-unit hidden layer was evaluated on the same frozen backbone representations.

| Benchmark | Metric | v1 | v2 | Δ |
|---|---|---:|---:|---:|
| STS-B | Spearman | 0.6996 | **0.7580** | **+0.0584** |
| AG News | Accuracy | 0.7700 | **0.8940** | **+0.1240** |
| SST-2 | Accuracy | 0.7420 | **0.8853** | **+0.1433** |
| MRPC | Accuracy | 0.7304 | **0.7721** | **+0.0417** |
| RTE | Accuracy | 0.6498 | **0.6679** | **+0.0181** |
| CoLA | Accuracy | 0.7110 | 0.7110 | 0.0000 |

Version 2 improves five of the six evaluated tasks under this nonlinear-probe configuration, matching the original on CoLA.

The contrasting linear and nonlinear results for RTE and MRPC illustrate an important property of representation evaluation: downstream performance depends not only on the information encoded by the backbone but also on the capacity of the mechanism used to recover it.

### Additional held-out evaluation

Following checkpoint selection, an additional evaluation was conducted using PAWS-Wiki and TREC coarse.

These datasets were not used to select the winning RAE-005 checkpoint.

| Dataset | Metric | v1 | v2 | Dispersion ON |
|---|---|---:|---:|---:|
| PAWS-Wiki | Accuracy | 0.6474 | **0.6855** | 0.6674 |
| TREC coarse | Accuracy | **0.9400** | 0.9360 | 0.9360 |

The evaluation included 8,000 PAWS-Wiki examples and 500 TREC examples.

On PAWS-Wiki, v2 improved absolute accuracy by 3.81 percentage points relative to v1 and outperformed the dispersion-treated checkpoint by 1.81 points.

On TREC coarse, both RAE-005 candidates scored 0.9360, slightly below the original checkpoint's 0.9400.

These observations supported the selected checkpoint under the experiment's predefined corroboration rule. They do not demonstrate uniformly improved generalization.

## Experimental finding: dispersion versus transfer

The RAE-005 experiment produced a distinction between training metrics, representation geometry, and downstream transfer.

At step 4,800:

| Measurement | Dispersion OFF | Dispersion ON |
|---|---:|---:|
| Training composite | 0.8280 | **0.8313** |
| STS-B Spearman improvement over v1 | **+0.0649** | +0.0140 |
| PAWS-Wiki accuracy | **0.6855** | 0.6674 |
| TREC coarse accuracy | 0.9360 | 0.9360 |

The treatment achieved a slightly higher training composite and reduced the reported terminal-layer representation-condensation measure by 0.0582.

The regularizer therefore produced a measurable change in the targeted representation geometry.

However, that geometric change did not correspond to the strongest downstream transfer under the RAE-005 protocol. The dispersion-free control performed substantially better on STS-B and PAWS-Wiki.

The earlier RAE-DISP-002 experiment had observed a small positive composite difference from terminal-layer dispersion in its cleaner matched-seed comparison (+0.0020). That finding and the RAE-005 result are not contradictory: they concern different training horizons, regularization locations, and evaluation objectives.

**The supported conclusion is bounded:** for this backbone, training configuration, and evaluation protocol, all-layer dispersion at λ = 0.1 did not outperform ordinary continuation on the selected transfer criteria.

The experiment does not establish that dispersion loss is generally ineffective, nor does it invalidate the findings reported by Liu et al. on other model and training configurations.

Representation geometry remains a useful diagnostic; it should not be substituted for downstream evaluation.

## How to use

RockAndEgg-v2 can be loaded using standard Hugging Face Transformers.

### Extract contextual representations

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "moeshawky/RockAndEgg-313M-v2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

texts = [
    "A stone has been painted to look like an egg.",
    "The decorated egg survived the contest."
]

batch = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=128,
    add_special_tokens=False,
    return_tensors="pt",
)

with torch.inference_mode():
    hidden = model(**batch).last_hidden_state

# Last non-padding position; robust to padding side.
mask = batch["attention_mask"].bool()
positions = torch.arange(
    mask.shape[1], device=mask.device
).unsqueeze(0).expand_as(mask)

last_idx = positions.masked_fill(~mask, -1).max(dim=1).values

features = hidden[
    torch.arange(hidden.shape[0], device=hidden.device),
    last_idx,
]

print(features.shape)  # [2, 1024]
```

This produces a contextual representation of each input suitable for a downstream trained head.

The example follows the release's last-non-padding-token feature-extraction convention. It does not imply that unadapted cosine similarity between the resulting vectors is a calibrated semantic similarity measure.

### Attach a classification head

A downstream classifier can be initialized using the Transformers sequence-classification interface:

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

model_id = "moeshawky/RockAndEgg-313M-v2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=3,
    problem_type="single_label_classification",
)
```

A new classification head is initialized and must be trained on the downstream task. Its initial predictions are not meaningful.

For reproducible comparison with the published frozen-probe results, use `AutoModel` and the specified pooling and probe protocol rather than assuming that full-model fine-tuning produces equivalent measurements.

## Limitations and considerations

**No pretrained decision head.** The checkpoint does not directly classify text into a predefined label space. Any task head must be trained and evaluated separately.

**Causal attention.** RockAndEgg retains the original causal attention mechanism. Its contextual representations are not equivalent to those of a bidirectional BERT-style encoder.

**Limited sequence-length evidence.** The configuration supports up to 32,768 positions, but the reported continuation training and transfer evaluation use a maximum of 128 tokens. Long-context performance remains unestablished.

**English-centric evaluation.** Although the Qwen3 tokenizer supports multilingual text, this release has not demonstrated competitive multilingual representation quality.

**Non-uniform transfer.** Improvements on STS-B, AG News, SST-2, and PAWS-Wiki coexist with regressions or unchanged results on other tasks.

**Probe dependence.** Linear and nonlinear probes recover different amounts of task-relevant information. Neither should be interpreted as a universal measure of backbone quality.

**Limited statistical replication.** The matched RAE-005 comparison provides controlled evidence for the tested interventions, but it does not establish a multi-seed confidence interval for the final transfer advantage.

**No independent architecture comparison.** Performance relative to BERT, DeBERTa, and other compact transformers remains to be established through like-for-like evaluation.

**Specialized deployments.** Medical, legal, employment, financial, and other high-impact applications require independent testing, error analysis, and calibration. Public benchmark performance is insufficient to certify suitability.

The released model also inherits potential biases and limitations associated with its Qwen3 donor and public supervised training sources.

## Reproducibility

### Model and research artifacts

| Artifact | Reference |
|---|---|
| Released model | [RockAndEgg-313M-v2](https://huggingface.co/moeshawky/RockAndEgg-313M-v2) |
| Original release | [RockAndEgg-313M](https://huggingface.co/moeshawky/RockAndEgg-313M) |
| Upstream donor | [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |
| Research evidence | [RAE-005 evidence archive](https://huggingface.co/datasets/moeshawky/rae-005-research-evidence) |

Reported immutable model revision:

`d738c1fcecf1c864901d747b1df1c79d9d7905ad`

Reported `model.safetensors` SHA-256:

`191d9597f60bb1350b66d7d1f5bd71410c716c6cd005334f4e61bc10f40a713b`

Research-evidence revision:

`6cf7d67c607f1f0a661b43b0a06afc666c5da236`

Original Qwen3 donor revision:

`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`

The release record reports independent clean-room weight retrieval, hash validation, successful `AutoModel` loading, finite forward outputs, and verification of the headless architecture.

The research archive contains the reported experiment protocols, qualification results, nonlinear-probe evaluation, held-out evaluation, geometry analysis, and associated provenance records.

### Reproduction boundaries

The RAE-005 comparison matched both experimental arms within the CUDA training environment.

Earlier TPU and CUDA dataset reconstruction paths were not established to be byte-identical, even where source datasets and row counts matched. Results across those earlier environments should not be interpreted as fully controlled comparisons without consulting their manifests.

Likewise, the original surgery experiments, RAE-TPU-003 continuation, earlier RAE-DISP trials, and final RAE-005 comparison represent successive research phases, not interchangeable benchmark runs.

## Research references

The following publications informed the project's architecture, experimental questions, or research context. Their roles are distinguished to avoid implying that every technique was implemented.

### Direct architectural and experimental foundations

**1. Qwen Team — Qwen3 Technical Report (2025)**

[arXiv:2505.09388](https://arxiv.org/abs/2505.09388)

The architectural and pretrained-weight foundation of RockAndEgg. The project retains ten physical transformer blocks from Qwen3-0.6B-Base and preserves the corresponding architectural interface.

**2. Liu et al. — Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models (ICML 2026)**

[Paper](https://proceedings.mlr.press/v306/liu26dy.html) · [Implementation](https://github.com/KrishnaswamyLab/LM-Dispersion)

The direct methodological basis for the experimental angular-dispersion intervention. Its published implementation informed the RAE-DISP studies and the RAE-005 treatment arm. The final released v2 checkpoint was selected without this auxiliary loss.

### Related research examined during development

**3. Wang et al. — MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers (NeurIPS 2020)**

[Paper](https://papers.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

Relevant to compact transformer design and teacher-student compression. RockAndEgg does not implement MiniLM's deep self-attention distillation objective.

**4. Wang et al. — MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers (ACL Findings 2021)**

[Paper](https://aclanthology.org/2021.findings-acl.188/)

Related work on relational knowledge transfer between transformer architectures. This method is not part of the selected v2 training recipe.

**5. Elhoushi et al. — LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding (ACL 2024)**

[Paper](https://aclanthology.org/2024.acl-long.681/)

Relevant background on useful representations at intermediate transformer depths. RockAndEgg uses physical depth truncation rather than LayerSkip's layer-dropout, early-exit, or speculative-decoding methods.

**6. Zhang et al. — Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models (2025)**

[arXiv:2506.05176](https://arxiv.org/abs/2506.05176)

Relevant Qwen-family work on representation learning and embedding/reranking specialization. RockAndEgg does not claim to reproduce its contrastive training pipeline, and the present release is not qualified as a dense retrieval model.

## Citation

If you use RockAndEgg-313M-v2 in research, please cite the model release and its Qwen3 foundation. If you discuss the dispersion experiment, also cite Liu et al. (2026).

```bibtex
@misc{rockandegg313mv2_2026,
  author       = {moeshawky},
  title        = {RockAndEgg-313M-v2: Multi-Task
                  Representation Learning with a
                  Truncated Qwen3 Backbone},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{
    https://huggingface.co/moeshawky/RockAndEgg-313M-v2
  }}
}
```

## License and attribution

RockAndEgg-313M-v2 is an independent derivative of Qwen3-0.6B-Base and is distributed under Apache License 2.0.

The original Qwen3 model, its authors, and their applicable license notices retain their attribution.

RockAndEgg is not affiliated with, sponsored by, or endorsed by Qwen or Alibaba.

## Project status

**RockAndEgg-313M-v2 is a completed foundation checkpoint, not the conclusion of the RockAndEgg research program.**

RAE-005 established a selected checkpoint, documented its transfer characteristics, and provided a controlled comparison of dispersion regularization under a specific training configuration.

The broader objective remains the development of a compact, competitive language-understanding backbone, to be judged against independently trained alternatives under fair downstream evaluation.

RockAndEgg remains an active applied model-development project.