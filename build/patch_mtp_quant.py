#!/usr/bin/env python3
"""Port the proven MTP quantized_layers remap into the image's mtp.py.

Root cause: vLLM builds the MTP draft model standalone with layer index 0,
but the checkpoint's quantized_layers list MTP experts under layer 48
(mtp.layers.48.mlp.experts, 0-based: start_layer 48). The image remaps only
ignored_layers/exclude_modules, so the draft quant config never sees
layer-48 prefixes -> MTP experts build UNQUANTIZED (bf16) -> the NVFP4
scale tensors (w13/w2_weight_scale_inv) have no parameter to land on ->
AttributeError at load.

This is the EXACT hunk the working vllm-fn4 venv has in
models/qwen4_exp/nvidia/mtp.py (two additions: the helper function and its
use in _make_draft_vllm_config). Idempotent; asserts anchors.
"""
import re
import py_compile

SP = "/usr/local/lib/python3.12/dist-packages/vllm"
F = f"{SP}/models/qwen3_8_flash_next/nvidia/mtp.py"

src = open(F, encoding="utf-8").read()
if "_remap_quantized_layers" in src:
    print("already patched")
    raise SystemExit(0)

# --- Hunk 1: helper function, right after _remap_ignored_layers -------------
helper_anchor = "    remapped.append(name)\n    return remapped\n"
assert src.count(helper_anchor) == 1, "helper anchor not unique"
helper = """    remapped.append(name)
    return remapped


def _remap_quantized_layers(
    quantized_layers: dict[str, dict],
    mtp_start_layer_idx: int,
) -> dict[str, dict]:
    \"\"\"Map checkpoint MTP layer indices to standalone draft indices.\"\"\"
    return {
        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info
        for name, layer_info in quantized_layers.items()
    }
"""
src = src.replace(helper_anchor, helper, 1)

# --- Hunk 2: remap quantized_layers inside _make_draft_vllm_config ----------
use_anchor = (
    '                "exclude_modules",\n'
    "                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),\n"
    "            )\n"
    "\n"
    "    draft_vllm_config = replace("
)
assert src.count(use_anchor) == 1, "use anchor not unique"
use = (
    '                "exclude_modules",\n'
    "                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),\n"
    "            )\n"
    "        quantized_layers = getattr(draft_quant_config, \"quantized_layers\", None)\n"
    "        if quantized_layers:\n"
    "            setattr(  # noqa: B010\n"
    "                draft_quant_config,\n"
    "                \"quantized_layers\",\n"
    "                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),\n"
    "            )\n"
    "\n"
    "    draft_vllm_config = replace("
)
src = src.replace(use_anchor, use, 1)

open(F, "w", encoding="utf-8").write(src)
py_compile.compile(F, doraise=True)
print("patched mtp.py (quantized_layers remap) — OK")
