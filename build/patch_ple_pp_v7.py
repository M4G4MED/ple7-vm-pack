#!/usr/bin/env python3
"""v7 patcher: cross-request state-corruption backports on top of ple-offload:v6.

EXPERIMENT-ONLY (image ple-offload:v6 -> v7; v6 stays bit-identical for A/B).
Target symptom (2026-09-25 log forensics): a RUNNING request suddenly pins to
0.0% draft acceptance at ALL positions + garbage/repetition output until it
finishes; other requests keep decoding fine; episodes began at the moment a
sibling request completed. Upstream threads: vllm#55357 (same model, same
fingerprint), #53912. This patcher carries the two merged upstream fixes whose
vulnerable code exists verbatim in our build (verified by anchor probes):

#25  v1/worker/gpu/spec_decode/{speculator.py,autoregressive/speculator.py,
     multi_module_mtp/speculator.py} + v1/worker/gpu/block_table.py
     — upstream PR #56734 (merged 2026-09-21, our wheel base is Sep-20).
     Dummy/draft-decode passes copy an arange idx_mapping into the drafter's
     persistent buffer; rows with idx_mapping == -1 (CUDA-graph padding) and
     every row of a dummy batch still index STALE request-state slots whose
     block tables belong to finished requests. The drafter then writes draft
     KV THROUGH THOSE STALE BLOCK TABLES — i.e. into blocks that were freed
     when a sibling request ended and may already be reused by a live one.
     Mechanism fits our fingerprint exactly: poisoned live request keeps
     generating with a draft KV group that no longer matches its own target
     state -> per-position acceptance flat 0.000 for the request's remaining
     life. Fix (a): a dummy batch fills idx_mapping with -1 everywhere;
     (b) the slot-mapping Triton kernel masks req_state_idx < 0 rows to PAD.

#26  v1/core/single_type_kv_cache_manager.py — upstream PR #55450
     (merged 2026-09-11; `_num_retired_blocks` absent from our build).
     Mamba "align" prefill leaves null gaps between state blocks awaiting
     retirement; the generic range-removal frees/rewrites across those gaps
     with stale indices, so states of still-running requests can be retired
     or overwritten out from under them (leak + stale-state reuse). Fix:
     MambaManager overrides _remove_blocks_in_range to walk gaps safely and
     remember per-request how far it already retired; bookkeeping is popped
     with the request.

Deliberately NOT included (verified not applicable / wrong class):
  - #58434 fixes a regression introduced by #52988 (Sep-23); our base predates
    #52988 — mamba_hybrid still classifies spec rows by num_scheduled_tokens,
    which handles padded prompt tails correctly. Anchors absent.
  - #58368 is a regression fix for #55390 (Sep-22, absent); the pre-55390
    fallback marks ALL groups eagle — conservative, hits shallower, no
    corruption. No drop_eagle_checkpoint_block attr to patch.
  - #54713/#53945 replay-boundary plumbing: prefix-cache HIT-RATE loss on
    block-aligned resends, not output corruption; large multi-file plumbing.
MTP config untouched (depth, method, drafts unchanged).
"""
import py_compile
import sys

PKG = "/usr/local/lib/python3.12/dist-packages/vllm"
SPC = f"{PKG}/v1/worker/gpu/spec_decode/speculator.py"
ARS = f"{PKG}/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
MMS = f"{PKG}/v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py"
BLT = f"{PKG}/v1/worker/gpu/block_table.py"
STK = f"{PKG}/v1/core/single_type_kv_cache_manager.py"


def edit(path, pairs, marker):
    src = open(path).read()
    if marker in src:
        print(f"[ple-pp-v7] {path.rsplit('/',1)[-1]} already patched")
        return
    for i, (old, new) in enumerate(pairs):
        n = src.count(old)
        assert n == 1, f"{path} anchor #{i} count={n}"
        src = src.replace(old, new)
    src = f"# ple-pp-v7\n" + src  # idempotency marker
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"[ple-pp-v7] patched + compiled {path.rsplit('/',1)[-1]} ({len(pairs)} edits)")


# ---------------- #25a: PR #56734 — drafter must not act on dummy rows -------
edit(SPC, [
    (
        """        # [max_num_reqs]
        seeds: torch.Tensor,
    ) -> None:
        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.""",
        """        # [max_num_reqs]
        seeds: torch.Tensor,
        dummy_run: bool = False,
    ) -> None:
        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.""",
    ),
    (
        """        self.idx_mapping[:num_reqs].copy_(idx_mapping)
        # idx_mapping for CG padded requests points to -1, which is ignored
        # during sampling to prevent writing stale values to draft logits.
        self.idx_mapping[num_reqs:].fill_(-1)""",
        """        # ple-pp-v7 #25 (upstream PR #56734): idx_mapping == -1 marks a row
        # the drafter must not act on: sampling skips it and
        # compute_slot_mappings emits PAD. CUDA-graph padded rows always get
        # it; a dummy batch gets it for every row, since its arange idx_mapping
        # names request-state slots with stale block tables (draft KV would be
        # written THROUGH those stale tables into freed/foreign blocks).
        if dummy_run:
            self.idx_mapping.fill_(-1)
        else:
            self.idx_mapping[:num_reqs].copy_(idx_mapping)
            self.idx_mapping[num_reqs:].fill_(-1)""",
    ),
], "ple-pp-v7")

# ---------------- #25b: thread dummy_run through both propose() callers ------
edit(ARS, [
    (
        """        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )""",
        """        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
            # ple-pp-v7 #25 (upstream PR #56734)
            dummy_run=dummy_run,
        )""",
    ),
], "ple-pp-v7")

edit(MMS, [
    (
        """        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )""",
        """        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
            # ple-pp-v7 #25 (upstream PR #56734)
            dummy_run=dummy_run,
        )""",
    ),
], "ple-pp-v7")

# ---------------- #25c: PR #56734 — slot-mapping kernel guards -1 rows -------
edit(BLT, [
    (
        """    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)""",
        """    req_state_idx = tl.load(idx_mapping + batch_idx)
    # ple-pp-v7 #25 (upstream PR #56734): idx_mapping == -1 marks a dummy (or
    # CUDA-graph padding) request that owns no blocks: never read its
    # block-table row and emit PAD for its tokens.
    is_real_req = req_state_idx >= 0
    start_idx = tl.load(query_start_loc + batch_idx)""",
    ),
    (
        """        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )""",
        """        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=is_real_req,
            other=0,
        )""",
    ),
    (
        """        slot_ids = tl.where(mapping_enabled, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)""",
        """        # ple-pp-v7 #25 (upstream PR #56734): fold is_real_req into the gate.
        slot_ids = tl.where(mapping_enabled & is_real_req, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)""",
    ),
], "ple-pp-v7")

# ---------------- #26: PR #55450 — retire Mamba states across null gaps ------
edit(STK, [
    (
        """            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}""",
        """            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            # ple-pp-v7 #26 (upstream PR #55450): how far range-removal has
            # already walked for this request, so null gaps between align-mode
            # prefill states are never re-processed with stale indices.
            self._num_retired_blocks: dict[str, int] = {}""",
    ),
    (
        """    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        assert isinstance(self.kv_cache_spec, MambaSpec)""",
        """    def _remove_blocks_in_range(
        self, request_id: str, first_block: int, last_block: int
    ) -> None:
        # ple-pp-v7 #26 (upstream PR #55450): Mamba "align" prefill leaves null
        # gaps between states awaiting retirement; walk them backward and
        # remember the retirement watermark instead of letting the generic
        # range-removal stale-index across the gaps.
        if self.mamba_cache_mode != "align":
            return super()._remove_blocks_in_range(request_id, first_block, last_block)
        blocks = self.req_to_blocks.get(request_id, [])
        first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))
        last_block = min(last_block, len(blocks))
        if first_block >= last_block:
            return
        freed = []
        # Mamba prefill leaves null gaps between states awaiting retirement.
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i].is_null:
                continue
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)
        self._num_retired_blocks[request_id] = last_block

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        assert isinstance(self.kv_cache_spec, MambaSpec)""",
    ),
    (
        """        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)""",
        """        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
            # ple-pp-v7 #26 (upstream PR #55450): pop the retirement watermark
            # with the request.
            self._num_retired_blocks.pop(request_id, None)""",
    ),
], "ple-pp-v7")

print("[ple-pp-v7] all patches applied")
sys.exit(0)
