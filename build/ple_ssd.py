"""SSD-backed PLE ngram table (page-cache offload) for the PLE CPU worker.

Active only when VLLM_PLE_SSD_OFFLOAD=1, inside the PleOffloadWorker
process (patched v1/ple_offload/worker.py calls prepare()/minimal_weights()/
finalize_load() there).

Without this module the worker materializes the full FP8 ngram table
(~47.7 GiB for Qwen3.8-Flash-Next) as anonymous host RAM. With it, the
table parameter's storage is a MAP_SHARED mmap over a file on fast SSD:

  * first launch: the file is created sparse at exact size; the NORMAL
    checkpoint stream's copy_ple_embedding_shard_() writes go straight
    into file pages. After load: fsync + header-fingerprint .meta;
  * later launches: fingerprint + 64-row byte-sample validated -> the
    loader stream is replaced by minimal_weights(), reading ONLY the tiny
    side tensors (weight_scale + 3 buffers, <200 bytes) named in
    model.safetensors.index.json. The 47.7 GiB of shard bytes is never
    opened; table "load" costs milliseconds, not ~6 minutes;
  * steady-state RAM = kernel page cache of hot ngram rows, reclaimable
    under memory pressure (unlike anon/tmpfs-backed offload).

Safety:
  * the storage swap happens BEFORE any copy, between layer construction
    and load_weights() (params are untouched torch.empty there);
  * reuse requires: .meta digest == sha256(safetensors header bytes),
    source size match, exact table size, and 64 deterministic
    pseudo-random checkpoint rows byte-compared against the table
    (catches all-zero/partial tables from interrupted builds);
  * FP8 shards must be byte-contiguous from offset 0 with one shared
    embedding dim (verified in _fingerprint) => file offset == runtime
    byte offset; vocab padding rows stay zero (stale files are rebuilt
    from sparse zeros);
  * a first build whose streamed tensor count != n_shards + n_side does
    not get a .meta (next launch rebuilds).

Env:
  VLLM_PLE_SSD_TABLE_PATH  table file (default /ple-ssd/ple_ngram.table);
                           MUST live on an uncompressed local filesystem.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import time

_TABLE_DEFAULT = "/ple-ssd/ple_ngram.table"
_HEADER_NAME = "model-fp8-mtp-ple.safetensors"
_INDEX_NAME = "model.safetensors.index.json"
_NGRAM_MARK = ".ngram_embedding."
_PLE_MARK = ".ple_embedding."
_SHARD_MARK = ".shard_"

_state: dict = {}


def _logger():
    from vllm.logger import init_logger

    return init_logger("vllm.ple_ssd")


def _fingerprint(source: str) -> dict:
    """Structure-only fingerprint + shard map of the PLE checkpoint file."""
    with open(source, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header_bytes = fh.read(header_len)
    header = json.loads(header_bytes)

    shards: dict[str, tuple[int, int]] = {}
    cols: set[int] = set()
    scale: str | None = None
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype, shape, offs = meta["dtype"], meta["shape"], meta["data_offsets"]
        if _NGRAM_MARK in name and _SHARD_MARK in name and name.endswith(".weight"):
            if dtype != "F8_E4M3" or len(shape) != 2:
                raise RuntimeError(
                    f"PLE SSD: unexpected shard {name} dtype={dtype} shape={shape}"
                )
            shards[name] = (int(offs[0]), int(offs[1]))
            cols.add(int(shape[1]))
        elif name.endswith(_NGRAM_MARK + "weight_scale"):
            scale = name
    if not shards or scale is None:
        raise RuntimeError(f"PLE SSD: FP8 ngram shards/scale not found in {source}")
    if len(cols) != 1:
        raise RuntimeError(f"PLE SSD: inconsistent shard cols {cols}")
    col = cols.pop()

    ordered = sorted((lo, hi, name) for name, (lo, hi) in shards.items())
    if any(ordered[i][1] != ordered[i + 1][0] for i in range(len(ordered) - 1)):
        raise RuntimeError("PLE SSD: FP8 shards are not byte-contiguous")
    start, end = ordered[0][0], ordered[-1][1]
    span = end - start
    if span != sum(hi - lo for lo, hi in shards.values()):
        raise RuntimeError("PLE SSD: shard span contains gaps")
    if span % col:
        raise RuntimeError(f"PLE SSD: table bytes {span} not divisible by dim {col}")
    # Nothing may sit between the first and last shard byte, or the
    # file-offset == runtime-byte-offset identity breaks.
    for name, meta in header.items():
        if name == "__metadata__" or name in shards:
            continue
        lo, hi = meta["data_offsets"]
        if start < lo < end or start < hi <= end:
            raise RuntimeError(
                f"PLE SSD: tensor {name} interleaved inside shard span"
            )

    return {
        "digest": hashlib.sha256(header_bytes).hexdigest(),
        "source": os.path.abspath(source),
        "source_size": os.path.getsize(source),
        "data_start": 8 + header_len,
        "source_offset": ordered[0][0],   # absolute file offset of row 0
        "file_bytes": end - ordered[0][0],
        "file_rows": (end - ordered[0][0]) // col,
        "cols": col,
        "n_shards": len(shards),
    }


def _model_dir(model_config) -> str:
    """Local checkpoint dir, mirroring DefaultModelLoader._prepare_weights,
    which is called with model_config.model and load_config.model_weights
    (the latter overrides only for special load formats like sharded-state)."""
    return getattr(model_config, "model_weights", "") or model_config.model


def _side_tensors(model_config) -> dict[str, str]:
    """Checkpoint-name -> file for non-shard PLE tensors (scale, buffers)."""
    index_path = os.path.join(_model_dir(model_config), _INDEX_NAME)
    try:
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"PLE SSD: cannot read {_INDEX_NAME} ({exc})") from exc
    # All non-shard tensors under the .ple_embedding. subtree: the FP8
    # weight_scale plus the persistent buffers (layer_multipliers,
    # ngram_heads_vocab_sizes, ngram_heads_offsets) — exactly what the
    # shard-free stream must replay for load_weights().
    side = {
        name: file
        for name, file in weight_map.items()
        if _PLE_MARK in name and _SHARD_MARK not in name
    }
    if not any(n.endswith(_NGRAM_MARK + "weight_scale") for n in side):
        raise RuntimeError("PLE SSD: index has no ngram weight_scale tensor")
    return side


def prepare(model_config, offload_layers: dict) -> None:
    """Swap every PLE weight's storage for a MAP_SHARED view of the table.

    Runs after layer construction (params are untouched torch.empty on
    CPU) and before model.load_weights().
    """
    log = _logger()
    if _state:
        return

    source = os.path.join(_model_dir(model_config), _HEADER_NAME)
    if not os.path.exists(source):
        raise RuntimeError(f"PLE SSD offload: checkpoint file missing: {source}")
    fp = _fingerprint(source)

    # Runtime table shape (vocab padding may exceed the checkpoint rows).
    shapes = {}
    for layer_name, layer in offload_layers.items():
        emb = getattr(layer, "ngram_embedding", None)
        w = getattr(emb, "weight", None) if emb is not None else None
        if w is None:
            raise RuntimeError(
                f"PLE SSD offload: {layer_name} has no ngram_embedding.weight"
            )
        if w.device.type != "cpu":
            raise RuntimeError("PLE SSD offload: expected CPU-owned PLE weight")
        shapes[layer_name] = tuple(w.shape)
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"PLE SSD offload: non-uniform table shapes {shapes}")
    rows, cols = next(iter(shapes.values()))
    if cols != fp["cols"]:
        raise RuntimeError(
            f"PLE SSD offload: runtime cols {cols} != checkpoint cols {fp['cols']}"
        )
    if rows < fp["file_rows"]:
        raise RuntimeError(
            f"PLE SSD offload: runtime rows {rows} < checkpoint rows "
            f"{fp['file_rows']}"
        )
    nbytes = rows * cols

    try:
        side = _side_tensors(model_config)
    except RuntimeError as exc:
        # Without the index map we cannot rebuild a skip-path stream;
        # still do the mmap (streamed copies land on SSD) but never skip.
        log.warning("%s — table stream will be required every launch.", exc)
        side = {}

    table = os.environ.get("VLLM_PLE_SSD_TABLE_PATH", _TABLE_DEFAULT)
    meta_path = table + ".meta"
    os.makedirs(os.path.dirname(table) or ".", exist_ok=True)

    skip = False
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
        skip = (
            bool(side)
            and meta.get("digest") == fp["digest"]
            and meta.get("bytes") == nbytes
            and os.path.getsize(table) == nbytes
        )
    except (OSError, ValueError):
        if os.path.exists(table):
            log.warning(
                "PLE SSD offload: table present but no valid .meta — this "
                "launch streams the checkpoint and rebuilds %s.",
                table,
            )

    fd = os.open(table, os.O_RDWR | os.O_CREAT, 0o644)
    if skip and os.fstat(fd).st_size != nbytes:
        skip = False
    if not skip:
        # Clean sparse slate: no stale bytes under padding, exact size.
        os.ftruncate(fd, 0)
        os.ftruncate(fd, nbytes)
    mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED,
                   prot=mmap.PROT_READ | mmap.PROT_WRITE)

    if skip:
        _sample_verify(mm, fp, table)
        log.info(
            "PLE SSD offload: table VALID (fingerprint + 64-row byte sample): "
            "%s — checkpoint stream replaced by %d tiny side tensors; %.1f "
            "GiB of shard reads avoided.",
            table, len(side), nbytes / 2**30,
        )

    import torch

    view = (
        torch.frombuffer(mm, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .view(rows, cols)
    )
    for layer_name, layer in offload_layers.items():
        w = layer.ngram_embedding.weight
        w.data = view  # param identity kept; storage becomes the mmap
    _state.update(
        mm=mm, fd=fd, skip=skip, nbytes=nbytes, fp=fp, meta_path=meta_path,
        rows=rows, cols=cols, table=table, side=side,
    )

    # 160-byte random gathers: suppress kernel readahead (MADV_RANDOM = 1;
    # 4 would be MADV_DONTNEED and would drop shared file pages!).
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.madvise(ctypes.c_void_p(view.data_ptr()),
                        ctypes.c_size_t(nbytes), 1) != 0:  # MADV_RANDOM
            log.debug("PLE SSD offload: madvise errno=%d", ctypes.get_errno())
    except Exception:
        pass

    log.info(
        "PLE SSD offload: attached %d PLE layer(s) to %s (%.1f GiB); "
        "steady-state table RAM is reclaimable page cache.",
        len(offload_layers), table, nbytes / 2**30,
    )


def _sample_verify(mm, fp: dict, table: str) -> None:
    """Byte-compare 64 deterministic pseudo-random checkpoint rows against
    the table. Catches all-zero/stale/partial tables a digest can't see."""
    import random

    col = fp["cols"]
    rng = random.Random(0xC0FFEE ^ fp["n_shards"])
    rows = sorted({rng.randrange(fp["file_rows"]) for _ in range(64)})
    base = fp["data_start"] + fp["source_offset"]
    src_fd = os.open(fp["source"], os.O_RDONLY)
    try:
        for r in rows:
            off = r * col
            want = os.pread(src_fd, col, base + off)
            if mm[off : off + col] != want:
                raise RuntimeError(
                    f"PLE SSD offload: table row {r} != checkpoint — delete "
                    f"{table} and {table}.meta, relaunch to rebuild."
                )
    finally:
        os.close(src_fd)


def minimal_weights(model_config):
    """Yield ONLY the tiny PLE side tensors (scale + persistent buffers).

    Used when the table is validated: the FP8 shard bytes are never
    opened. Names are original HF names — exactly what the real loader
    would have yielded.
    """
    log = _logger()
    model_path = _model_dir(model_config)
    side: dict[str, str] = _state["side"]
    from safetensors import safe_open

    by_file: dict[str, list[str]] = {}
    for name, file in side.items():
        by_file.setdefault(file, []).append(name)
    started = time.time()
    total = 0
    for file in sorted(by_file):
        with safe_open(os.path.join(model_path, file), framework="pt",
                       device="cpu") as fh:
            for name in sorted(by_file[file]):
                tensor = fh.get_tensor(name)
                total += tensor.numel() * tensor.element_size()
                yield name, tensor
    log.info(
        "PLE SSD offload: side-tensor stream done — %d tensor(s), %.2f MiB "
        "in %.1f s (0 shard bytes read).",
        len(side), total / 2**20, time.time() - started,
    )


def skipping() -> bool:
    return bool(_state.get("skip"))


def finalize_load(matched_checkpoint_tensors: int) -> None:
    """After a successful first build: count-check, fsync, persist .meta."""
    if not _state:
        return
    log = _logger()
    if _state["skip"]:
        log.info("PLE SSD offload: reused validated table; nothing to persist.")
        return
    fp = _state["fp"]
    if not _state["side"]:
        log.warning(
            "PLE SSD offload: index map unavailable — cannot verify build "
            "completeness; not writing .meta (next launch streams again)."
        )
        return
    expected = fp["n_shards"] + len(_state["side"])
    if matched_checkpoint_tensors != expected:
        log.error(
            "PLE SSD offload: streamed %d tensors, expected %d (%d shards + "
            "%d side) — NOT marking table complete; next launch rebuilds.",
            matched_checkpoint_tensors, expected, fp["n_shards"],
            len(_state["side"]),
        )
        return
    started = time.time()
    _state["mm"].flush()
    os.fsync(_state["fd"])
    with open(_state["meta_path"] + ".tmp", "w") as fh:
        json.dump(
            {
                "digest": fp["digest"],
                "source_size": fp["source_size"],
                "bytes": _state["nbytes"],
                "rows": _state["rows"],
                "cols": _state["cols"],
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            fh,
        )
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(_state["meta_path"] + ".tmp", _state["meta_path"])
    log.info(
        "PLE SSD offload: table built+fsynced in %.1f min; next launches "
        "skip the %.1f GiB checkpoint table stream.",
        (time.time() - started) / 60.0, _state["nbytes"] / 2**30,
    )
