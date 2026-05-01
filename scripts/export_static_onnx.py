"""
Export a HuggingFace embedding model to ONNX with static shapes for QNN NPU compatibility.

QNN EP cannot handle dynamic shapes (Shape ops with dynamic dims cause full CPU fallback).
This script exports with fixed batch_size=1 and seq_len, then simplifies the graph
to fold constants and remove Shape ops.

Usage:
    python scripts/export_static_onnx.py --model BAAI/bge-base-en-v1.5 --seq-len 128 --output models/bge-base-en-v1.5-static
    python scripts/export_static_onnx.py --model sentence-transformers/all-MiniLM-L6-v2 --seq-len 128 --output models/all-MiniLM-L6-v2-static
"""

import argparse
import json
import os
import sys

# Support pylibs install location
pylibs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pylibs")
if os.path.isdir(pylibs):
    sys.path.insert(0, pylibs)

import torch
import onnx
from transformers import AutoModel, AutoTokenizer


def export_static(model_name: str, seq_len: int, output_dir: str, opset: int = 17):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    batch_size = 1
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    has_token_type_ids = "token_type_ids" in tokenizer.model_input_names

    if has_token_type_ids:
        dummy_inputs = (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
    else:
        dummy_inputs = (dummy_input_ids, dummy_attention_mask)
        input_names = ["input_ids", "attention_mask"]

    output_path = os.path.join(output_dir, "model.onnx")

    print(f"Exporting to ONNX (opset={opset}, batch=1, seq_len={seq_len}, static shapes)...")
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=["last_hidden_state"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"Exported: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")

    # Simplify to fold remaining constants and remove Shape ops
    try:
        import onnxsim
        print("Simplifying ONNX model...")
        model_onnx = onnx.load(output_path)
        model_simplified, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simplified, output_path)
            print(f"Simplified: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
        else:
            print("WARNING: Simplification validation failed, keeping original")
    except ImportError:
        print("onnxsim not installed, skipping simplification")

    # Verify no Shape/Erf ops remain
    model_onnx = onnx.load(output_path)
    ops = set(n.op_type for n in model_onnx.graph.node)
    problematic = {"Shape", "Erf"} & ops
    if problematic:
        print(f"WARNING: Problematic ops still present: {problematic}")
    else:
        print("No Shape/Erf ops found - should be QNN compatible!")

    for inp in model_onnx.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param:
                dims.append(d.dim_param + " (DYNAMIC!)")
            else:
                dims.append(str(d.dim_value))
        print(f"  Input '{inp.name}': [{', '.join(dims)}]")

    print(f"Total ops: {len(model_onnx.graph.node)}, unique: {sorted(ops)}")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    tokenizer_json = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json):
        print(f"Tokenizer saved to {output_dir}")
    else:
        print("WARNING: No tokenizer.json found - may need manual copy")

    # Create model_config.json with fixed max_tokens
    config = {"max_tokens": seq_len}
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path} (max_tokens={seq_len})")

    print(f"\nDone! Model exported to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HuggingFace model to static-shape ONNX for QNN NPU")
    parser.add_argument("--model", required=True, help="HuggingFace model name (e.g., BAAI/bge-base-en-v1.5)")
    parser.add_argument("--seq-len", type=int, default=128, help="Fixed sequence length (default: 128)")
    parser.add_argument("--output", required=True, help="Output directory for the static ONNX model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    export_static(args.model, args.seq_len, args.output, args.opset)
