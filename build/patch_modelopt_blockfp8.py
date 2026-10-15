#!/usr/bin/env python3
"""Port the proven block-FP8 MoE support into the image's modelopt.py.

Root cause of:
  AttributeError: Layer mtp.layers.48.mlp.experts has no parameter
  'w2_weight_scale_inv' for checkpoint weight 'mtp.layers.48.mlp.experts.0.
  down_proj.weight_scale_inv'

Your checkpoint's MTP experts are block-FP8 (hf_quant_config.json:
mtp.layers.0.mlp.experts -> FP8_BLOCK_SCALES group 128; real tensors carry
weight_scale_inv). The 48 main experts are NVFP4 — the MTP layer is the ONLY
block-FP8 MoE. The image's ModelOptMixedPrecisionConfig.get_quant_method has
no RoutedExperts branch for FP8_BLOCK_SCALES -> returns None -> experts build
unquantized -> the scale tensors have no param to land on.

This is the EXACT local patch the working vllm-fn4 venv has in
model_executor/layers/quantization/modelopt.py (diff vs its .orig-55513
backup): Fp8MoEMethod branch for block-FP8 MoE algos, the fp8_block_config
built in __init__, the has_blocked_weights gate (enables +quant_fp8 compile
op — referenced by config/vllm.py via hasattr), and the algo alias constant.
Idempotent; anchors asserted; py_compile at the end.
"""
import py_compile

SP = "/usr/local/lib/python3.12/dist-packages/vllm"
F = f"{SP}/model_executor/layers/quantization/modelopt.py"

src = open(F, encoding="utf-8").read()
if "_BLOCK_FP8_MOE_ALGOS" in src:
    print("already patched")
    raise SystemExit(0)

# --- Hunk 1: import Fp8Config/Fp8MoEMethod ----------------------------------
h1_old = (
    "    QuantizeMethodBase,\n"
    ")\n"
    "from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod\n"
)
h1_new = (
    "    QuantizeMethodBase,\n"
    ")\n"
    "from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod\n"
    "from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod\n"
)
assert src.count(h1_old) == 1, "hunk1 anchor not unique"
src = src.replace(h1_old, h1_new, 1)

# --- Hunk 2: algo alias constant (after the logger) --------------------------
h2_old = "logger = init_logger(__name__)\n"
h2_new = (
    "logger = init_logger(__name__)\n"
    "\n"
    "# ``FP8_PB_WO`` is ModelOpt's canonical 2D block-FP8 name. Early composed\n"
    "# Qwen3.8-Flash-Next checkpoints used ``FP8_BLOCK_SCALES`` for the same tensor\n"
    "# layout, so retain it as a checkpoint-compatibility alias.\n"
    "_BLOCK_FP8_MOE_ALGOS = (\"FP8_PB_WO\", \"FP8_BLOCK_SCALES\")\n"
)
assert src.count(h2_old) == 1, "hunk2 anchor not unique"
src = src.replace(h2_old, h2_new, 1)

# --- Hunks 3+4: __init__ tail (fp8_block_config) + has_blocked_weights -------
h3_old = (
    "        self.w4a16_nvfp4_config = w4a16_nvfp4_config\n"
    "        self.mxfp8_config = mxfp8_config\n"
)
h3_new = (
    "        self.w4a16_nvfp4_config = w4a16_nvfp4_config\n"
    "        self.mxfp8_config = mxfp8_config\n"
    "\n"
    "        block_sizes = {\n"
    "            int(layer_info.get(\"group_size\", 128))\n"
    "            for layer_info in quantized_layers.values()\n"
    "            if layer_info.get(\"quant_algo\", \"\").upper() in _BLOCK_FP8_MOE_ALGOS\n"
    "        }\n"
    "        if len(block_sizes) > 1:\n"
    "            raise ValueError(\n"
    "                \"MIXED_PRECISION currently requires all block-FP8 MoE layers \"\n"
    "                f\"to use one group_size, got {sorted(block_sizes)}.\"\n"
    "            )\n"
    "        block_size = next(iter(block_sizes), 128)\n"
    "        self.fp8_block_config = Fp8Config(\n"
    "            is_checkpoint_fp8_serialized=True,\n"
    "            activation_scheme=\"dynamic\",\n"
    "            weight_block_size=[block_size, block_size],\n"
    "        )\n"
    "\n"
    "    def has_blocked_weights(self) -> bool:\n"
    "        # \"quant_fp8\" must be on as soon as any layer is block-scaled\n"
    "        # (same gate as ModelOptFp8Config.has_blocked_weights, resolved\n"
    "        # per layer).\n"
    "        return any(\n"
    "            info.get(\"quant_algo\", \"\").upper() in _BLOCK_FP8_MOE_ALGOS\n"
    "            for info in self.quantized_layers.values()\n"
    "        )\n"
)
assert src.count(h3_old) == 1, "hunk3 anchor not unique"
src = src.replace(h3_old, h3_new, 1)

# --- Hunk 5: RoutedExperts branch in ModelOptMixedPrecisionConfig ------------
# Confine to the class region so we can't hit another class's get_quant_method.
class_off = src.find("class ModelOptMixedPrecisionConfig")
assert class_off > 0
region_off = src.find("def get_quant_method", class_off)
assert region_off > class_off
branch_old = (
    "        if isinstance(layer, RoutedExperts):\n"
    "            if quant_algo == \"FP8\":\n"
)
branch_new = (
    "        if isinstance(layer, RoutedExperts):\n"
    "            if quant_algo in _BLOCK_FP8_MOE_ALGOS:\n"
    "                return Fp8MoEMethod(self.fp8_block_config, layer)\n"
    "            if quant_algo == \"FP8\":\n"
)
assert src[region_off:region_off + 4000].count(branch_old) == 1, "hunk5 anchor not unique in class"
head = src[:region_off]
rest = src[region_off:]
rest = rest.replace(branch_old, branch_new, 1)
src = head + rest

open(F, "w", encoding="utf-8").write(src)
py_compile.compile(F, doraise=True)
print("patched modelopt.py (block-FP8 MoE: Fp8MoEMethod branch) — OK")
