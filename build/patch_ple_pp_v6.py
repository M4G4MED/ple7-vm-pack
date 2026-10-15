#!/usr/bin/env python3
"""v6 patcher: agentic-stability backports on top of ple-offload:v5.

EXPERIMENT-ONLY (image ple-offload:v6; v5/v4 stacks stay bit-identical).
Every patch is an upstream/community fix for SILENT corruption in exactly our
config (PP2 + MTP + prefix caching + hybrid GDN), plus per-rank KV budgets.
MTP itself is untouched (depth, method, drafts all unchanged).

#21  v1/worker/gpu/pp_utils.py   — upstream PR #54436
     compute_need_sampled_mask()'s `not_finishing` guard drops a decoding row
     once num_computed overruns prompt+max_tokens (spec decode advances it k
     tokens/step, so the guard misfires WHILE the request is still running).
     The earlier stage's last_sampled_tokens/draft_tokens then freeze while
     the last rank keeps advancing -> permanent ring drift -> the observed
     'ahahaha' garbage + acceptance collapse to 0. This is the root cause of
     the drift that patch #19 only detects and repairs; #21 removes it.
     Truly-finished reqs stay covered by req_idx_gen (get_prev_sampled_outputs
     already filters freed rows via gen_at_receive_np).

#22  v1/core/single_type_kv_cache_manager.py — upstream PR #57128
     MambaManager.find_longest_cache_hit accepts drop_eagle_block but ignores
     it: a cached GDN state written over REJECTED draft positions stays
     reachable and is reused by any later request sharing the prefix ->
     silent corruption (empty/repeated output, no error). Fix: when
     drop_eagle_block is set, skip exactly the FIRST (most recent) real match
     and keep scanning for the older committed checkpoint. Must not pre-shrink
     the ceiling: in align mode checkpoints only exist at sparse block
     boundaries, so shrinking lands in a gap and zeroes the hit rate.

#23  v1/worker/gpu_worker.py     — zebgop 0014 (via vektorprime patches-v2)
     Per-rank KV budget override VLLM_KV_CACHE_MEMORY_RANK<i>. Under PP the
     ranks are heterogeneous (PLE machinery + layers 0-24 on rank 0, MTP
     drafter + layers 25-47 on rank 1); one global --kv-cache-memory fits the
     tightest rank and strands memory on the other. Inert unless the env is set.

#24  v1/worker/gpu_model_runner.py — upstream PR #53919 (V1-runner path)
     (a) _update_states must wait on num_accepted_tokens_event BEFORE the
     first row move: the non-blocking D2H of accepted counts lands in a pinned
     buffer that add_request/condense permute -> stale/foreign accepted count
     -> wrong GDN snapshot selector -> cross-request state bleed.
     (b) The async prev_positions double-permute gather re-permutes values the
     event wait already made correct -> one request gets another's accepted
     count. Direct read in every mode. Our stack runs MRV2 so this is a
     belt-and-braces backport (vektorprime kept it for exactly that reason);
     it costs nothing on the V2 path.
"""
import py_compile
import sys

PKG = "/usr/local/lib/python3.12/dist-packages/vllm"
PPU = f"{PKG}/v1/worker/gpu/pp_utils.py"
STK = f"{PKG}/v1/core/single_type_kv_cache_manager.py"
GWK = f"{PKG}/v1/worker/gpu_worker.py"
MR1 = f"{PKG}/v1/worker/gpu_model_runner.py"


def edit(path, pairs, marker):
    src = open(path).read()
    if marker in src:
        print(f"[ple-pp-v6] {path.rsplit('/',1)[-1]} already patched")
        return
    for i, (old, new) in enumerate(pairs):
        n = src.count(old)
        assert n == 1, f"{path} anchor #{i} count={n}"
        src = src.replace(old, new)
    src = f"# ple-pp-v6\n" + src  # idempotency marker
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"[ple-pp-v6] patched + compiled {path.rsplit('/',1)[-1]} ({len(pairs)} edits)")


# ---------------- #21: PR #54436 — never drop a decoding row ----------------
edit(PPU, [
    (
        """    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    max_seq_len = input_batch.max_seq_len_np
    assert max_seq_len is not None  # always populated under PP
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # Exclude requests that we know are finished.
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None""",
        """    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # ple-pp-v6 #21 (upstream PR #54436): the old `not_finishing` guard is
    # REMOVED. Under spec decode num_computed overruns prompt+max_tokens
    # while the scheduler still runs the request; dropping its row froze the
    # earlier stage's last_sampled_tokens/draft_tokens -> permanent ring
    # drift (our 'ahahaha' incident, acceptance 0). Truly-finished requests
    # are already covered by the req_idx_gen filter in get_prev_sampled_outputs.
    return produces_sample if produces_sample.any() else None""",
    ),
], "ple-pp-v6")

# ---------------- #22: PR #57128 — mamba respects drop_eagle_block ----------
edit(STK, [
    (
        """            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )
            for fine_idx in range(max_num_partial_units - 1, -1, -1):
                num_tokens = (fine_idx + 1) * hash_block_size
                block_hash = block_hashes[fine_idx]
                if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids
                ):
                    block_idx = fine_idx // scale_factor""",
        """            max_num_partial_units = min(
                max_length // hash_block_size, len(block_hashes)
            )
            # ple-pp-v6 #22 (upstream PR #57128): honor drop_eagle_block in
            # align mode. Real checkpoints exist only at sparse boundaries,
            # so we must NOT pre-shrink the ceiling (lands in a gap -> 0%
            # hits); instead skip exactly the FIRST real match (the risky one
            # that may hold state written over rejected MTP drafts) and keep
            # scanning for the older committed checkpoint below it.
            skip_next_hit = drop_eagle_block
            for fine_idx in range(max_num_partial_units - 1, -1, -1):
                num_tokens = (fine_idx + 1) * hash_block_size
                block_hash = block_hashes[fine_idx]
                if cached_block := block_pool.get_cached_block(
                    block_hash, kv_cache_group_ids
                ):
                    if skip_next_hit:
                        skip_next_hit = False
                        continue
                    block_idx = fine_idx // scale_factor""",
    ),
    (
        """            return computed_blocks, hit_length

        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue""",
        """            return computed_blocks, hit_length

        max_num_blocks = max_length // block_size
        # ple-pp-v6 #22 (upstream PR #57128): see fine-grained branch above —
        # skip only the FIRST real (alignment-valid) match when
        # drop_eagle_block is set, then keep scanning for the next one.
        skip_next_hit = drop_eagle_block
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                if skip_next_hit:
                    skip_next_hit = False
                    continue""",
    ),
], "ple-pp-v6")

# ---------------- #23: zebgop 0014 — per-rank KV budgets --------------------
edit(GWK, [
    (
        """        maybe_apply_startup_plan(self)
""",
        """        maybe_apply_startup_plan(self)

        # ple-pp-v6 #23 (zebgop 0014): per-rank KV budget override. Under PP
        # the ranks are heterogeneous (PLE machinery on rank 0, MTP drafter on
        # the last rank), so one global value must fit the tightest rank and
        # strands memory everywhere else. Inert unless VLLM_KV_CACHE_MEMORY_RANK<i>
        # is set.
        import os as _os
        _pp_rank = get_pp_group().rank_in_group
        _override = _os.getenv(f"VLLM_KV_CACHE_MEMORY_RANK{_pp_rank}")
        if _override and not self.cache_config.kv_cache_memory_bytes:
            self.cache_config.kv_cache_memory_bytes = int(_override)
            logger.info(
                "Per-rank KV budget override: pp_rank=%d takes %s bytes from "
                "VLLM_KV_CACHE_MEMORY_RANK%d",
                _pp_rank, _override, _pp_rank,
            )
""",
    ),
], "ple-pp-v6")

# ---------------- #24: PR #53919 — accepted-count race (V1 path) -------------
edit(MR1, [
    (
        '''    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
''',
        '''    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
        # ple-pp-v6 #24 (upstream PR #53919 part 1): the accepted-token counts
        # copied D2H (non-blocking) by step N's mamba postprocess land in a
        # pinned buffer that this step's row moves (add_request/swap_states/
        # condense) permute. Wait BEFORE the first row move so the DMA has
        # landed; syncing an unrecorded event is a no-op.
        if self.num_accepted_tokens_event is not None:
            self.num_accepted_tokens_event.synchronize()
''',
    ),
    (
        """            # Async mode: condense() reordered indices, use prev_positions mapping
            if self.use_async_scheduling and prev_req_id_to_index:
                prev_idx = self.prev_positions.np[:num_reqs]
                new_mask = prev_idx < 0
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[
                        np.where(new_mask, 0, prev_idx)
                    ]
                )
                self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
                self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                    self.num_accepted_tokens.np[:num_reqs]
                )
            else:
                # Non-async mode: use values directly
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )""",
        """            # ple-pp-v6 #24 (upstream PR #53919 part 2): the prev_positions
            # double-permute gather is removed. The event wait at the top of
            # _update_states guarantees the copy landed before condense()/
            # swap_states ran, so the buffer is already in CURRENT row order;
            # the old gather re-permuted already-correct values whenever the
            # DMA landed early -> one request gets another's accepted count
            # -> wrong GDN snapshot selector -> cross-request state bleed.
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )""",
    ),
], "ple-pp-v6")

print("[ple-pp-v6] all patches applied")
sys.exit(0)
