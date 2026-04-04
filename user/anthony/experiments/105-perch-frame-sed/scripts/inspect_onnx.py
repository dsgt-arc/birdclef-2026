"""Download Perch v2 ONNX and dump a structural report.

Purpose: task 1.1 / 1.2 of exp 105. We need to understand the Perch v2 ONNX
graph well enough to (a) identify the pre-pool feature-map tensor we want
to tap, and (b) confirm we can run the model end-to-end before doing any
graph surgery.

Outputs:
  - stdout: human-readable report
  - artifacts/onnx_inspection/summary.json: machine-readable metadata
  - artifacts/onnx_inspection/nodes.txt: every node (op_type, name, inputs, outputs)
  - artifacts/onnx_inspection/candidates.txt: pooling-adjacent nodes — likely tap points

Run via sbatch/inspect-onnx.sbatch, not on the login node.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO_ID = "justinchuby/Perch-onnx"
# Same filename used by exp 104 — broader onnxruntime compatibility than the
# DFT variant.
FILENAME = "perch_v2_no_dft.onnx"

ARTIFACT_DIR = Path("artifacts/onnx_inspection")


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def dtype_name(elem_type: int) -> str:
    return onnx.TensorProto.DataType.Name(elem_type)


def shape_of(value_info) -> list:
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    banner("1. Download Perch v2 ONNX")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  repo      : {REPO_ID}")
    print(f"  filename  : {FILENAME}")
    print(f"  cached at : {model_path}")
    print(f"  size      : {size_mb:.1f} MB")

    banner("2. Load ONNX graph")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    graph = model.graph
    print(f"  ir_version  : {model.ir_version}")
    print(f"  producer    : {model.producer_name} {model.producer_version}")
    print(f"  opset       : {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    print(f"  graph name  : {graph.name}")
    print(f"  nodes       : {len(graph.node)}")
    print(f"  inputs      : {len(graph.input)}")
    print(f"  outputs     : {len(graph.output)}")
    print(f"  initializers: {len(graph.initializer)}")

    banner("3. Inputs")
    inputs_summary = []
    for i, vi in enumerate(graph.input):
        dt = dtype_name(vi.type.tensor_type.elem_type)
        sh = shape_of(vi)
        print(f"  [{i}] {vi.name:<40} dtype={dt:<8} shape={sh}")
        inputs_summary.append({"name": vi.name, "dtype": dt, "shape": sh})

    banner("4. Outputs")
    outputs_summary = []
    for i, vi in enumerate(graph.output):
        dt = dtype_name(vi.type.tensor_type.elem_type)
        sh = shape_of(vi)
        print(f"  [{i}] {vi.name:<40} dtype={dt:<8} shape={sh}")
        outputs_summary.append({"name": vi.name, "dtype": dt, "shape": sh})

    banner("5. Op-type histogram")
    op_counts: dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    for op, count in sorted(op_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>6}  {op}")

    banner("6. Parameter count (from initializers)")
    total_params = 0
    for init in graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        total_params += n
    print(f"  total initializer elements: {total_params:,}")
    print(f"  approx (fp32 MB)          : {total_params * 4 / 1024 / 1024:.1f}")

    banner("7. Last 40 nodes (tail of graph — where the head lives)")
    tail_start = max(0, len(graph.node) - 40)
    for idx, node in enumerate(graph.node[tail_start:], start=tail_start):
        name = node.name or "<anon>"
        print(f"  [{idx:>5}] {node.op_type:<20} name={name}")
        print(f"          in : {list(node.input)}")
        print(f"          out: {list(node.output)}")

    banner("8. Pool / reduction / flatten candidates (likely tap neighbours)")
    pool_ops = {
        "GlobalAveragePool",
        "GlobalMaxPool",
        "AveragePool",
        "ReduceMean",
        "ReduceMax",
        "Flatten",
        "Squeeze",
        "Gemm",
        "MatMul",
    }
    candidates = []
    for idx, node in enumerate(graph.node):
        if node.op_type in pool_ops:
            candidates.append((idx, node))
    for idx, node in candidates:
        name = node.name or "<anon>"
        print(f"  [{idx:>5}] {node.op_type:<20} name={name}")
        print(f"          in : {list(node.input)}")
        print(f"          out: {list(node.output)}")

    banner("9. Dummy forward pass (sanity check)")
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        feeds = {}
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            print(f"  feeding {inp.name}: shape={shape} dtype={inp.type}")
            if "float" in inp.type:
                feeds[inp.name] = np.zeros(shape, dtype=np.float32)
            elif "int64" in inp.type:
                feeds[inp.name] = np.zeros(shape, dtype=np.int64)
            else:
                feeds[inp.name] = np.zeros(shape, dtype=np.float32)
        out_names = [o.name for o in sess.get_outputs()]
        outs = sess.run(out_names, feeds)
        for name, arr in zip(out_names, outs):
            print(f"  output {name}: shape={arr.shape} dtype={arr.dtype}")
    except Exception as e:
        print(f"  FORWARD PASS FAILED: {type(e).__name__}: {e}")
        print("  (graph inspection above is still valid)")

    banner("10. Write artifacts")
    summary = {
        "repo_id": REPO_ID,
        "filename": FILENAME,
        "model_path": str(model_path),
        "size_mb": size_mb,
        "ir_version": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}",
        "opset": [(o.domain or "ai.onnx", o.version) for o in model.opset_import],
        "num_nodes": len(graph.node),
        "num_initializers": len(graph.initializer),
        "total_params": total_params,
        "op_counts": op_counts,
        "inputs": inputs_summary,
        "outputs": outputs_summary,
        "pool_candidate_indices": [idx for idx, _ in candidates],
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {ARTIFACT_DIR / 'summary.json'}")

    with (ARTIFACT_DIR / "nodes.txt").open("w") as fh:
        for idx, node in enumerate(graph.node):
            fh.write(
                f"[{idx:>5}] {node.op_type:<24} name={node.name or '<anon>'}\n"
                f"         in : {list(node.input)}\n"
                f"         out: {list(node.output)}\n"
            )
    print(f"  wrote {ARTIFACT_DIR / 'nodes.txt'}")

    with (ARTIFACT_DIR / "candidates.txt").open("w") as fh:
        for idx, node in candidates:
            fh.write(
                f"[{idx:>5}] {node.op_type:<24} name={node.name or '<anon>'}\n"
                f"         in : {list(node.input)}\n"
                f"         out: {list(node.output)}\n"
            )
    print(f"  wrote {ARTIFACT_DIR / 'candidates.txt'}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
