#!/usr/bin/env python3
"""
Test suite for Qwen3-313M v2 conversions.

Tests every conversion against the base checkpoint at:
  /kaggle/input/models/moeshawky/rockandegg-313m-v2/transformers/default/1

Test categories:
  1. Structural: each artifact loads, correct architecture/tensor count/types
  2. Numerical fidelity: FP16 vs F32 tensors
  3. Quantization error: GGUF tensors dequantized vs F32 original tensors
  4. AutoRound: int4/int8 safetensors structure + config vs F32 (packed format,
     tensor-level dequantization requires AutoRound's inference path)

No transformers import — pure tensor comparison via safetensors + gguf.
Run: python3 test_conversions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, dequantize
from safetensors.torch import load_file as load_sf

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_CKPT = "/kaggle/input/models/moeshawky/rockandegg-313m-v2/transformers/default/1"
FP16_CKPT = "/kaggle/working/hive-classifier-detour-20261115/timing/deps/fp16"
GGUF_DIR = Path("/root/quantize-work")
AUTOROUND_DIR = Path("/root/quantize-work")

TOLERANCE = {
    "fp16_tensor": 2e-2,      # FP16 tensor vs F32 original (norm weights are small)
    "gguf_tensor": 2.0,       # Dequantized GGUF tensor vs F32 original (aggressive quant)
}

results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "details": [],
}


def record(name: str, ok: bool, detail: str = ""):
    if ok:
        results["passed"] += 1
        status = "PASS"
    else:
        results["failed"] += 1
        status = "FAIL"
    results["details"].append(f"  [{status}] {name}: {detail}")
    print(f"  [{status}] {name}: {detail}")


# ── Load base F32 safetensors ──────────────────────────────────────────────
print("\n=== Loading base F32 safetensors ===")
f32_path = f"{BASE_CKPT}/model.safetensors"
f32_state = load_sf(f32_path, device="cpu")
print(f"  Loaded {len(f32_state)} tensors from base F32 checkpoint")

# ── Tensor name mapping (GGUF -> safetensors) ──────────────────────────────
n_layers = 10
sf_to_gguf = {
    "embed_tokens.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
for i in range(n_layers):
    sf_to_gguf.update({
        f"layers.{i}.input_layernorm.weight": f"blk.{i}.attn_norm.weight",
        f"layers.{i}.post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
        f"layers.{i}.self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
        f"layers.{i}.self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
        f"layers.{i}.self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
        f"layers.{i}.self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
        f"layers.{i}.self_attn.q_norm.weight": f"blk.{i}.attn_q_norm.weight",
        f"layers.{i}.self_attn.k_norm.weight": f"blk.{i}.attn_k_norm.weight",
        f"layers.{i}.mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
        f"layers.{i}.mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
        f"layers.{i}.mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
    })


def dequantize_gguf_tensor(gtensor) -> np.ndarray:
    """Dequantize a GGUF ReaderTensor to float32 numpy array."""
    data = gtensor.data
    qtype = gtensor.tensor_type
    if int(qtype) in (1, 30):
        return np.asarray(data, dtype=np.float32)
    return dequantize(data, qtype).astype(np.float32)


# ── 1. FP16 safetensors ───────────────────────────────────────────────────
print("\n=== FP16 safetensors (timing/deps/fp16/) ===")
fp16_path = f"{FP16_CKPT}/model.safetensors"
fp16_state = load_sf(fp16_path, device="cpu")
record("FP16 tensor count", len(fp16_state) == 112, f"count={len(fp16_state)}")

fp16_dtypes = {str(v.dtype) for v in fp16_state.values()}
record("FP16 all tensors are float16", fp16_dtypes == {"torch.float16"},
       f"dtypes={fp16_dtypes}")

fp16_max_diff = 0.0
fp16_worst = ""
for sf_name, f32_tensor in f32_state.items():
    if sf_name not in fp16_state:
        continue
    fp16_tensor = fp16_state[sf_name].numpy().astype(np.float32)
    f32_np = f32_tensor.numpy().astype(np.float32)
    diff = float(np.abs(fp16_tensor - f32_np).max())
    if diff > fp16_max_diff:
        fp16_max_diff = diff
        fp16_worst = sf_name

record("FP16 max_abs_diff vs F32",
       fp16_max_diff < TOLERANCE["fp16_tensor"],
       f"{fp16_max_diff:.3e} (tensor={fp16_worst}, tolerance {TOLERANCE['fp16_tensor']})")

# ── 2. GGUF files ──────────────────────────────────────────────────────────
print("\n=== GGUF tensor quantization error vs F32 original ===")
for gguf_name in ["model-f16.gguf", "model-q4km.gguf", "model-q4ks.gguf", "model-q80.gguf"]:
    gguf_path = GGUF_DIR / gguf_name
    if not gguf_path.exists():
        record(f"{gguf_name} exists", False, "file not found")
        continue

    record(f"{gguf_name} exists", True,
           f"size={gguf_path.stat().st_size / 1e6:.1f} MB")

    reader = GGUFReader(str(gguf_path), mode="r")
    gguf_tensors = {t.name: t for t in reader.tensors}
    record(f"{gguf_name} tensor count", len(gguf_tensors) == 112,
           f"count={len(gguf_tensors)}")

    # Check architecture
    arch_found = False
    for field in reader.fields.values():
        if field.name == "general.architecture":
            contents = field.contents()
            arch_val = contents[0] if len(contents) == 1 else contents
            arch_found = str(arch_val) == "qwen3"
    record(f"{gguf_name} architecture=qwen3", arch_found, "arch=qwen3")

    # Compare each mapped tensor
    max_diff = 0.0
    worst_tensor = ""
    compared = 0
    for gname, gtensor in gguf_tensors.items():
        if gname not in sf_to_gguf.values():
            continue
        # Find the safetensors name for this GGUF tensor
        sname = None
        for sf, gg in sf_to_gguf.items():
            if gg == gname:
                sname = sf
                break
        if sname is None or sname not in f32_state:
            continue

        f32_np = f32_state[sname].numpy().astype(np.float32)
        gdata = dequantize_gguf_tensor(gtensor)

        # Handle shape/layout differences (GGUF may store transposed)
        if gdata.shape != f32_np.shape:
            if len(gdata.shape) == 2 and len(f32_np.shape) == 2:
                gdata_t = gdata.T
                if gdata_t.shape == f32_np.shape:
                    gdata = gdata_t
            if gdata.shape != f32_np.shape:
                gdata = gdata.flatten()
                f32_np = f32_np.flatten()

        diff = float(np.abs(gdata - f32_np).max())
        if diff > max_diff:
            max_diff = diff
            worst_tensor = gname
        compared += 1

    record(f"{gguf_name} tensors compared", compared > 0,
           f"compared={compared}")
    record(f"{gguf_name} max_abs_diff vs F32",
           max_diff < TOLERANCE["gguf_tensor"],
           f"{max_diff:.3e} (tensor={worst_tensor}, tolerance {TOLERANCE['gguf_tensor']})")

# ── 3. AutoRound conversions ───────────────────────────────────────────────
print("\n=== AutoRound conversions ===")
for label, subdir, expected_bits in [("int4", "fp16-w4g128", 4), ("int8", "fp16-w8g128", 8)]:
    ar_path = AUTOROUND_DIR / f"qwen3-313m-{label}-autoround" / subdir
    safetensors_path = ar_path / "model.safetensors"
    config_path = ar_path / "quantization_config.json"
    if ar_path.exists() and safetensors_path.exists():
        record(f"AutoRound {label} output exists", True, f"at {ar_path}")

        # Check quantization config
        if config_path.exists():
            with open(config_path) as f:
                ar_cfg = json.load(f)
            cfg_bits = ar_cfg.get("bits")
            cfg_group = ar_cfg.get("group_size")
            cfg_sym = ar_cfg.get("sym")
            record(f"AutoRound {label} config bits={expected_bits}",
                   cfg_bits == expected_bits,
                   f"bits={cfg_bits}, group_size={cfg_group}, sym={cfg_sym}")
        else:
            record(f"AutoRound {label} quantization config exists", False,
                   "config not found")

        # Check tensor structure
        ar_state = load_sf(str(safetensors_path), device="cpu")
        ar_dtypes = {str(v.dtype) for v in ar_state.values()}
        has_qweight = any("qweight" in k for k in ar_state.keys())
        record(f"AutoRound {label} has packed qweight tensors", has_qweight,
               f"dtypes={sorted(ar_dtypes)}, count={len(ar_state)}")

        # Count weight tensors (those without .qweight/.qzeros/.scales suffix)
        base_names = set()
        for k in ar_state.keys():
            base = k.rsplit(".", 1)[0] if "." in k else k
            base_names.add(base)
        record(f"AutoRound {label} logical weight count", len(base_names) == 112,
               f"logical weights={len(base_names)}")
    else:
        record(f"AutoRound {label} output exists", False, "not yet complete")
        results["skipped"] += 1

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"TEST SUMMARY: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped")
print("=" * 60)
for d in results["details"]:
    print(d)

sys.exit(0 if results["failed"] == 0 else 1)
