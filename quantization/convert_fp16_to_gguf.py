#!/usr/bin/env python3
"""Convert Qwen3-313M HF safetensors checkpoint (FP16 storage) to GGUF F16.

Reads model.safetensors from the FP16 checkpoint dir, writes a self-contained
GGUF file with all required Qwen3 metadata and F16 tensors. Output goes to
/root/quantize-work/ (963 GB available) since /kaggle/working is tight (19 GB).

This is the baseline F16 GGUF from which Q4_K_M, Q4_K_S, Q8_0 are derived
via llama-quantize.

Usage:
  python3 convert_fp16_to_gguf.py <safetensors_dir> <output_gguf>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter, RopeScalingType


CKPT = "/kaggle/working/hive-classifier-detour-20261115/timing/deps/fp16"
OUT = "/root/quantize-work/model-f16.gguf"


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(CKPT)
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(OUT)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Load config ──
    config = json.loads((ckpt / "config.json").read_text())
    print(f"Config architecture: {config['architectures']}")
    print(f"  hidden_size={config['hidden_size']} layers={config['num_hidden_layers']}")
    print(f"  heads={config['num_attention_heads']} kv_heads={config['num_key_value_heads']}")
    print(f"  head_dim={config['head_dim']} intermediate={config['intermediate_size']}")
    print(f"  vocab={config['vocab_size']} rope_theta={config['rope_parameters']['rope_theta']}")

    n_layers = config["num_hidden_layers"]
    dim = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    ffn_dim = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    rope_theta = config["rope_parameters"]["rope_theta"]
    rms_norm_eps = config["rms_norm_eps"]
    max_ctx = config.get("max_position_embeddings", 32768)

    # ── Load safetensors ──
    from safetensors.torch import load_file

    sf_path = ckpt / "model.safetensors"
    state = load_file(str(sf_path), device="cpu")
    print(f"Loaded {len(state)} tensors from {sf_path.name}")

    # ── Build GGUF writer ──
    # architecture set in constructor ("qwen3")
    fout = GGUFWriter(str(output), "qwen3")

    # General metadata
    fout.add_name("Qwen3-313M-FP16-GGUF")
    fout.add_vocab_size(vocab_size)
    fout.add_context_length(max_ctx)
    fout.add_embedding_length(dim)
    fout.add_feed_forward_length(ffn_dim)
    fout.add_expert_count(0)  # dense model
    fout.add_block_count(n_layers)
    fout.add_head_count(n_heads)
    fout.add_head_count_kv(n_kv_heads)
    fout.add_layer_norm_rms_eps(rms_norm_eps)
    fout.add_key_length(head_dim)
    fout.add_value_length(head_dim)
    fout.add_rope_dimension_count(head_dim)
    fout.add_rope_freq_base(rope_theta)
    fout.add_rope_scaling_type(RopeScalingType.LINEAR)
    fout.add_file_type(1)  # F16

    # Tensor mapping: safetensors name -> GGUF name
    # Checkpoint uses names without "model." prefix: layers.{i}.self_attn.q_proj.weight, etc.
    tensor_map = {
        "embed_tokens.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }

    for i in range(n_layers):
        tensor_map.update({
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

    # Add tensors in F16
    added = 0
    skipped = 0
    for sf_name, tensor in state.items():
        gguf_name = tensor_map.get(sf_name)
        if gguf_name is None:
            gguf_name = sf_name
            skipped += 1
            print(f"  UNMAPPED: {sf_name} -> {gguf_name}")
            continue

        data = tensor.numpy().astype(np.float16)
        fout.add_tensor(gguf_name, data)
        added += 1

    print(f"Added {added} tensors, skipped {skipped}")

    # ── Write ──
    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()

    size_gb = output.stat().st_size / 1e9
    print(f"\nWrote {output} ({size_gb:.2f} GB)")
    print("DONE")


if __name__ == "__main__":
    main()
