#!/usr/bin/env python3
"""v4 patcher: SSD-backed (page-cache) PLE ngram table for the cross-process
CPU offload worker (VLLM_PLE_CPU_OFFLOAD=1 + VLLM_PLE_SSD_OFFLOAD=1).

Problem: the PLE CPU worker materializes the ~47.7 GiB FP8 ngram table as
anonymous host RAM on a 62 GiB box.

Fix (helper module vllm/v1/ple_offload/ple_ssd.py, COPYed in by Dockerfile):
back the table param's storage with a MAP_SHARED mmap over a file on fast
SSD, swapped in BEFORE any copy happens:
  * First launch: file created sparse; the normal checkpoint stream's
    copy_ple_embedding_shard_() writes (narrow().copy_() into
    embedding.weight.data, re-read per shard) land straight in file pages.
    fsync + header-fingerprint .meta after a complete build.
  * Later launches: fingerprint + 64-row byte-sample validated -> the whole
    loader stream is replaced by minimal_weights(): only the tiny side
    tensors (weight_scale + buffers) named in model.safetensors.index.json
    are read; the 47.7 GiB shard file is never opened ("load" in ms).
  * Steady-state RAM = kernel page cache of hot ngram rows — reclaimable
    under memory pressure (unlike anon pages / pinned tmpfs).

No CUDA-IPC interaction: the table never crosses processes (only small GPU
output buffers do; that path was the expandable_segments/pidfd_getfd bug).
"""
import os
import sys

WP = "/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py"

src = open(WP).read()
if "ple_ssd" in src:
    print("[ple-ssd] worker.py already patched — nothing to do")
    sys.exit(0)

HELPER = "/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/ple_ssd.py"
if not os.path.exists(HELPER):
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ple_ssd.py")
    if os.path.exists(alt):
        open(HELPER, "w").write(open(alt).read())
        print(f"[ple-ssd] helper installed from {alt}")
    else:
        sys.exit("[ple-ssd] FATAL: helper ple_ssd.py not found")

# ---- A0) worker.py has no `import os`; patch B needs it --------------------
a0 = "import multiprocessing.process\n"
assert src.count(a0) == 1, "anchor A0"
src = src.replace(a0, "import multiprocessing.process\nimport os\n")

# ---- A) helper handle init before loader selection -------------------------
a = """        loader = get_model_loader(load_config)
        if isinstance(loader, DummyModelLoader):"""
assert src.count(a) == 1, "anchor A"
src = src.replace(
    a,
    """        _ple_ssd = None
        loader = get_model_loader(load_config)
        if isinstance(loader, DummyModelLoader):""",
)

# ---- B) attach mmap storage; skip shard stream on validated table ----------
b = """        elif isinstance(loader, DefaultModelLoader):
            all_weights = loader.get_all_weights(model_config, model)
            loaded_params = model.load_weights(offload_only_iter(all_weights))"""
assert src.count(b) == 1, "anchor B"
src = src.replace(
    b,
    """        elif isinstance(loader, DefaultModelLoader):
            if os.environ.get("VLLM_PLE_SSD_OFFLOAD", "0") == "1":
                from vllm.v1.ple_offload import ple_ssd as _ple_ssd

                # Swaps every PLE weight's storage for a MAP_SHARED view of
                # the SSD table file BEFORE any copy happens: first-run
                # streamed writes land on disk, validated runs read page
                # cache and skip the shard stream entirely.
                _ple_ssd.prepare(model_config, offload_layers)
                if _ple_ssd.skipping():
                    all_weights = _ple_ssd.minimal_weights(model_config)
                else:
                    all_weights = loader.get_all_weights(model_config, model)
            else:
                all_weights = loader.get_all_weights(model_config, model)
            loaded_params = model.load_weights(offload_only_iter(all_weights))""",
)

# ---- C) the shard stream is skipped BY DESIGN on validated tables ----------
c = """            loaded_expected_params = expected_offload_params.intersection(loaded_params)
            missing_offload_params = sorted("""
assert src.count(c) == 1, "anchor C"
src = src.replace(
    c,
    """            if _ple_ssd is not None and _ple_ssd.skipping():
                # The SSD table already holds the FP8 rows (fingerprint +
                # byte-sample validated); shard tensors were intentionally
                # not streamed — credit the table parameter as loaded.
                loaded_params = set(loaded_params) | {
                    name
                    for name in expected_offload_params
                    if name.endswith(".weight")
                }
            loaded_expected_params = expected_offload_params.intersection(loaded_params)
            missing_offload_params = sorted(""",
)

# ---- D) fsync + fingerprint after a successful first build ------------------
d = """        for layer in offload_layers.values():
            process_weights_after_loading(layer, model_config, torch.device("cpu"))

        self._layers.update(offload_layers)"""
assert src.count(d) == 1, "anchor D"
src = src.replace(
    d,
    """        for layer in offload_layers.values():
            process_weights_after_loading(layer, model_config, torch.device("cpu"))

        if _ple_ssd is not None:
            # First run: count-check, fsync the table, write the fingerprint
            # .meta so the next launch takes the skip path.
            _ple_ssd.finalize_load(matched_checkpoint_tensors)

        self._layers.update(offload_layers)""",
)

open(WP, "w").write(src)
import py_compile

py_compile.compile(WP, doraise=True)
py_compile.compile(HELPER, doraise=True)
print("[ple-ssd] worker.py patched, worker+helper compile OK")
