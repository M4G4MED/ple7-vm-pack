#!/usr/bin/env python3
"""Patch vllm PLE gate in the official qwen38-flash-next image to accept
ModelOptMixedPrecisionConfig checkpoints (NVFP4 experts + FP8 PLE table).

Inserts exactly the branch that the proven local vllm-fn4 venv uses
(models/qwen4_exp/nvidia/ple_layer.py:139-142), so this checkpoint's PLE
table loads as FP8 (47.7 GiB) instead of BF16 (~95 GiB) and dequantizes
with its checkpoint scale on the CPU offload path.

Idempotent: safe to re-run; anchors asserted so a drifted upstream file
fails loudly instead of patching wrong.
"""
import sys

F = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
src = open(F).read()

ANCHOR_IMP = "from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n"
ANCHOR_GATE = (
    "    if not isinstance(quant_config, Fp8Config):\n"
    "        return None\n"
)
BRANCH = (
    "    if isinstance(quant_config, ModelOptMixedPrecisionConfig):\n"
    "        # This checkpoint (Qwen3.8-Flash-Next NVFP4, ModelOpt\n"
    "        # MIXED_PRECISION: NVFP4 routed experts + FP8 PLE ngram table)\n"
    "        # serializes the PLE shards as per-tensor FP8. Route them to\n"
    "        # the FP8 embedding method so the table is created as FP8\n"
    "        # (~47.7 GiB, not ~95 GiB BF16) and the CPU-offload lookup\n"
    "        # dequantizes with the checkpoint's global scale.\n"
    "        if quant_config._resolve_quant_algo(prefix) == \"FP8\":\n"
    "            return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\n"
    "        return None\n"
)
IMP_LINE = (
    "from vllm.model_executor.layers.quantization.modelopt import (\n"
    "    ModelOptMixedPrecisionConfig,\n"
    ")\n"
)

if BRANCH not in src:
    assert ANCHOR_GATE in src, "GATE ANCHOR NOT FOUND - upstream file drifted, refusing to patch"
    if ANCHOR_IMP not in src:
        print("FATAL: import anchor not found - upstream file drifted", file=sys.stderr)
        sys.exit(2)
    if IMP_LINE not in src:
        src = src.replace(ANCHOR_IMP, ANCHOR_IMP + IMP_LINE, 1)
        print("added import line")
    src = src.replace(ANCHOR_GATE, BRANCH + ANCHOR_GATE, 1)
    print("inserted ModelOptMixedPrecisionConfig gate branch")
else:
    print("branch already present - no-op")
    if IMP_LINE not in src:
        print("FATAL: branch present but import missing - inconsistent state", file=sys.stderr)
        sys.exit(3)

open(F, "w").write(src)

import py_compile
py_compile.compile(F, doraise=True)
print("py_compile OK")

# post-check: the gate now references both classes
chk = open(F).read()
assert "isinstance(quant_config, ModelOptMixedPrecisionConfig)" in chk
assert "isinstance(quant_config, Fp8Config)" in chk
print("PATCHED OK")
