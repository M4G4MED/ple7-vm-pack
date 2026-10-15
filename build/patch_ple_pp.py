#!/usr/bin/env python3
"""v5 patcher: run PLE SSD offload + MTP under pipeline_parallel_size>=2.

EXPERIMENT-ONLY (image ple-offload:v5; the v4 stack stays bit-identical).
Three coordinated files, each grounded in this fork's own source and in
upstream vllm issue #49355 ("MTP speculative decoding is broken with
pipeline parallelism — three distinct failures"):

1) v1/worker/gpu_worker.py
   _validate_ple_offload_config rejects PP>1 by design. Allow PP only when
   VLLM_PLE_PP_EXPERIMENTAL=1 is set — existing v4 launchers never set it,
   so their behavior cannot change.

2) v1/ple_offload/connector.py
   A pipeline stage that owns no PleOffloadLayer (ple_layer_ids=[2] -> only
   the first stage, since make_layers materializes real modules only in
   [start_layer,end_layer)) becomes a PASSIVE connector: no CUDA-IPC
   registration, no ZMQ requests. The table gather is requested ONLY by the
   stage that executes the PLE placeholder, so the worker's
   single-request-per-step semaphore protocol remains exactly the validated
   TP2 behavior. A second-stage request would be silently dropped anyway
   (worker dedups by dp_rank) and desync the sem handshake — this keeps it
   from ever being sent.

3) v1/worker/gpu_model_runner.py
   self.drafter is created only on the last PP rank, but seven sites
   dereference it under a bare self.speculative_config truthiness gate
   (issue #49355 Failure 1: "'GPUModelRunner' object has no attribute
   'drafter'" during determine_available_memory / metadata build /
   _dummy_run / initialize_attn_backend / _check_and_update_cudagraph_mode).
   Guard each with get_pp_group().is_last_rank, matching the creation guard
   — this is the fix the issue author validated locally.

Failure 2 of #49355 (async scheduling PP assert sampled_token_ids [num_reqs,1]
vs MTP's [num_reqs,k+1]) is avoided operationally: the PP launcher passes
--no-async-scheduling. Failure 3 (logits gather OOB, reported on a Nemotron
Step3p5-MTP model) is the residual risk the bench must disprove; it has no
known a-priori patch.

worker.py needs NO change: spawn happens on world-rank 0 (TP0 of PP stage 0,
which owns the PLE layer), num_workers = dp*tp equals the number of stage-0
registrations, and the CPU worker builds its meta model under the isolated
PP1 gloo world, so it always sees the full 48-layer table regardless of the
serving PP size.

4) models/qwen3_8_flash_next/nvidia/model.py  (Qwen3_8FlashNextModel.load_weights)
   The final GatedResidual `hyper_connection_mixer` is built ONLY on the last
   PP stage (self.hyper_connection_mixer = None elsewhere), but the checkpoint
   streams its top-level columns to EVERY stage. A non-last stage therefore has
   no such module, and the AutoWeightsLoader raises
   "no module or parameter named 'hyper_connection_mixer'". On non-last stages
   drop the whole prefix; the last stage loads it exactly as PP=1 does.
   (embed_tokens is already a PPMissingLayer on non-first stages and lm_head a
   PPMissingLayer on non-last stages, so those are skipped automatically.)
"""
import py_compile
import sys

PKG = "/usr/local/lib/python3.12/dist-packages/vllm"
GW = f"{PKG}/v1/worker/gpu_worker.py"
CONN = f"{PKG}/v1/ple_offload/connector.py"
MR = f"{PKG}/v1/worker/gpu_model_runner.py"
MODEL = f"{PKG}/models/qwen3_8_flash_next/nvidia/model.py"


def edit(path, pairs):
    src = open(path).read()
    if "ple-pp-v5" in src:
        print(f"[ple-pp] {path.rsplit('/',1)[-1]} already patched")
        return
    for i, (old, new) in enumerate(pairs):
        n = src.count(old)
        assert n == 1, f"{path} anchor #{i} count={n}"
        src = src.replace(old, new)
    src = "# ple-pp-v5\n" + src  # idempotency marker (comment line)
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"[ple-pp] patched + compiled {path.rsplit('/',1)[-1]} ({len(pairs)} edits)")


# ---------------------------------------------------------------- 1) gpu_worker
edit(
    GW,
    [
        (
            """        if parallel_config.pipeline_parallel_size != 1:
            unsupported.append(f"PP={parallel_config.pipeline_parallel_size}")""",
            """        if (
            parallel_config.pipeline_parallel_size != 1
            and os.environ.get("VLLM_PLE_PP_EXPERIMENTAL", "0") != "1"
        ):
            unsupported.append(f"PP={parallel_config.pipeline_parallel_size}")""",
        ),
    ],
)

# ---------------------------------------------------------------- 2) connector
edit(
    CONN,
    [
        # passive stage: no PLE layer here -> empty layer set is legal
        (
            """        if not layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )""",
            """        if not layers:
            # PP>1: a stage without PLE placeholders never consumes CPU
            # results. Stay passive instead of failing the whole engine.
            # (PLE-offload launchers with PP=1 always have >=1 layer, so the
            # original error condition is preserved for them below.)
            logger.info(
                "PleOffload: no PleOffloadLayer on this pipeline stage; "
                "passive connector (no registration, no requests).",
            )
            return {}""",
        ),
        # skip registration/pinning/request-thread for a passive connector
        (
            """        try:
            self._zmq_ctx = zmq.Context()""",
            """        if not self._layers:
            return
        try:
            self._zmq_ctx = zmq.Context()""",
        ),
        # belt-and-braces: a passive stage must never emit requests
        (
            """        if self.tp_rank != 0:
            return

        if self._uses_cuda_inputs:""",
            """        if self.tp_rank != 0:
            return
        if not self._layers:
            # Passive pipeline stage: the stage owning the PLE layer drives
            # the gather; a second request would desync the sem protocol.
            return

        if self._uses_cuda_inputs:""",
        ),
    ],
)

# ------------------------------------------------- 3) gpu_model_runner (MTP+PP)
edit(
    MR,
    [
        # site 1: per-KV-group spec metadata builder
        (
            """            if self.speculative_config and spec_decode_common_attn_metadata is None:""",
            """            if (
                self.speculative_config
                and get_pp_group().is_last_rank
                and spec_decode_common_attn_metadata is None
            ):""",
        ),
        # site 2: Qwen MTP per-group block table
        (
            """            if self.speculative_config and isinstance(
                self.drafter, Qwen3_8FlashNextMTPProposer
            ):""",
            """            if (
                self.speculative_config
                and get_pp_group().is_last_rank
                and isinstance(self.drafter, Qwen3_8FlashNextMTPProposer)
            ):""",
        ),
        # site 3: Step3p5 MTP per-group attn metadata (elif of site 2)
        (
            """            elif self.speculative_config and isinstance(
                self.drafter, Step3p5MTPProposer
            ):""",
            """            elif (
                self.speculative_config
                and get_pp_group().is_last_rank
                and isinstance(self.drafter, Step3p5MTPProposer)
            ):""",
        ),
        # site 4: Gemma4 per-group block table (elif of the chain)
        (
            """            elif self.speculative_config and isinstance(self.drafter, Gemma4Proposer):""",
            """            elif (
                self.speculative_config
                and get_pp_group().is_last_rank
                and isinstance(self.drafter, Gemma4Proposer)
            ):""",
        ),
        # site 5: _dummy_run -> drafter.dummy_run (12-space indent block)
        (
            """            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):""",
            """            if (
                self.speculative_config
                and get_pp_group().is_last_rank
                and (
                    self.speculative_config.use_eagle()
                    or self.speculative_config.uses_draft_model()
                    or self.speculative_config.uses_extract_hidden_states()
                )
            ):""",
        ),
        # site 6: initialize_attn_backend drafter hook (8-space, 2 methods)
        (
            """        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
        ):""",
            """        if (
            self.speculative_config
            and get_pp_group().is_last_rank
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
            )
        ):""",
        ),
        # site 7: _check_and_update_cudagraph_mode drafter hook (8-space, 3 methods)
        (
            """        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_draft_model()
            or self.speculative_config.uses_extract_hidden_states()
        ):""",
            """        if (
            self.speculative_config
            and get_pp_group().is_last_rank
            and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            )
        ):""",
        ),
    ],
)

# ------------------------------------------- 13) model_state.py (PLE PP guard)
# Qwen3_8FlashNextModelState.__init__ hard-rejects PP>1 ("non-first ranks do not
# receive raw input_ids"). Under PP2 the ngram_context buffer is still built on
# every stage but is only CONSUMED by the PleOffloadLayer on stage 0 (PPMissing
# elsewhere), and PP ranks do get token ids replayed via pp_handler sampled-token
# broadcast — so the buffers exist and stay unused on stage 1. Gate the guard.
MODEL_STATE = f"{PKG}/models/qwen3_8_flash_next/nvidia/model_state.py"
edit(
    MODEL_STATE,
    [
        (
            """from typing import Any""",
            """import os
from typing import Any""",
        ),
        (
            """        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )""",
            """        if (
            vllm_config.parallel_config.pipeline_parallel_size > 1
            and os.environ.get("VLLM_PLE_PP_EXPERIMENTAL") != "1"
        ):
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )""",
        ),
    ],
)

# ------------------------------------------- 15) mtp.py (draft PP branch bug)
# The draft MTP forward carries the target model's PP boilerplate:
#   if get_pp_group().is_first_rank: <embed+fusion branch>
#   else: assert intermediate_tensors is not None
# But the draft (a) is only ever BUILT on the last PP rank (speculator gate),
# (b) is only ever CALLED by the speculator, which passes hidden_states +
# input_ids directly and never intermediate_tensors. On the last rank of a
# PP>1 world is_first_rank is False -> the assert always fires (verified live
# 2026-09-18: profile_run -> _dummy_run -> speculator.propose -> _prefill ->
# dynamo 'Data-dependent assertion failed', ConstantVariable(bool: False)).
# PP=1 is unaffected: rank 0 is first AND last, so the embed branch already
# runs in production. Discriminate on what the caller actually passed, gated
# by the experiment env so the PP=1 code path cannot change.
MTP = f"{PKG}/models/qwen3_8_flash_next/nvidia/mtp.py"
edit(
    MTP,
    [
        (
            """from collections.abc import Iterable""",
            """import os
from collections.abc import Iterable""",
        ),
        (
            """        if get_pp_group().is_first_rank:
            assert hidden_states is not None""",
            """        if (
            get_pp_group().is_first_rank
            or (
                os.environ.get("VLLM_PLE_PP_EXPERIMENTAL") == "1"
                and intermediate_tensors is None
            )
        ):
            assert hidden_states is not None""",
        ),
    ],
)

# --------------------------------------------------- 4) model.py (PP loader gap)
edit(
    MODEL,
    [
        (
            """        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]""",
            """        skip_substrs = [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]
        # PP>1: the final hyper_connection_mixer is built only on the last
        # stage (None elsewhere), but its top-level checkpoint columns are
        # streamed to every stage -> loader ValueError on non-last stages.
        # Drop the whole prefix there; the last stage loads it as in PP=1.
        if not get_pp_group().is_last_rank:
            skip_substrs.append("hyper_connection_mixer.")""",
        ),
    ],
)

# --------------------------------- 16) kv_cache_utils.py (PP diag dump, run 5)
# PP-only divergence: the engine core merges both stages' per-layer specs and
# demands one spec per mamba bucket; the bare ValueError hides WHICH field
# diverges (shapes vs page_size_padded vs mode vs num_spec_blocks). Dump the
# per-owner diff against the representative before raising — same error, plus
# the exact field values (rep=first-owner stage, this=diverging owner).
KVU = f"{PKG}/v1/core/kv_cache_utils.py"
edit(
    KVU,
    [
        (
            """        representative = roles.mamba[names[0]]
        if any(roles.mamba[name] != representative for name in names[1:]):
            policy = "replicated" if tp_replicated else "sharded"
            raise ValueError(
                f"CSA+linear {policy} mamba cache owners must use one spec."
            )""",
            """        representative = roles.mamba[names[0]]
        if any(roles.mamba[name] != representative for name in names[1:]):
            policy = "replicated" if tp_replicated else "sharded"
            # ple-pp-v5 diag: identify the diverging field + owner stage
            try:
                import dataclasses as _dc

                for _n in names:
                    _s = roles.mamba[_n]
                    _diff = {
                        _f.name: (
                            getattr(representative, _f.name),
                            getattr(_s, _f.name),
                        )
                        for _f in _dc.fields(representative)
                        if getattr(representative, _f.name)
                        != getattr(_s, _f.name)
                    }
                    logger.error(
                        "ple-pp mamba-dump policy=%s owner=%s "
                        "unpadded_page=%s diff(rep,this)=%s",
                        policy,
                        _n,
                        replace(_s, page_size_padded=None).page_size_bytes,
                        _diff,
                    )
            except Exception as _e:  # never mask the original error
                logger.error("ple-pp mamba-dump failed: %s", _e)
            raise ValueError(
                f"CSA+linear {policy} mamba cache owners must use one spec."
            )""",
        ),
    ],
)

# ------------------------------------------- 17) pp_utils.py (broadcast width)
# Flight-recorder verdict (run 3): receiver pre-allocates the sampled-token
# broadcast buffer as [num_reqs, max_sample_len] (=k+1=5), but the sender
# broadcasts the RAW sampler output — [4,1] during warmup (no draft metadata ->
# plain-sampler width 1). NCCL seq#0 on pp_broadcast group: 4x5 vs 4x1 desync
# (was the silent hang before TORCH_DISTRIBUTED_DEBUG=DETAIL; PP1's sampler is
# width-1 on any step whose drafts were all rejected/pre-draft too). Pad the
# sender to the width the receiver always reserves; -1 fill == the rejection
# sampler's own invalid-slot contract, so post_update semantics are unchanged.
PPU = f"{PKG}/v1/worker/gpu/pp_utils.py"
edit(
    PPU,
    [
        (
            """from collections import deque""",
            """import os
from collections import deque""",
        ),
        (
            """        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():""",
            """        assert sampled_token_ids.dtype == torch.int64
        # ple-pp-v5: keep broadcast width == receiver's reserved max_sample_len.
        if (
            os.environ.get("VLLM_PLE_PP_EXPERIMENTAL") == "1"
            and sampled_token_ids.shape[-1] != self.max_sample_len
        ):
            assert sampled_token_ids.shape[-1] < self.max_sample_len
            _padded = torch.full(
                (sampled_token_ids.shape[0], self.max_sample_len),
                -1,
                dtype=sampled_token_ids.dtype,
                device=sampled_token_ids.device,
            )
            _padded[:, : sampled_token_ids.shape[-1]] = sampled_token_ids
            sampled_token_ids = _padded

        if current_platform.is_xpu():""",
        ),
        # ---- 18) Draft-token relay (backport of upstream main pp_utils).
        # Root cause of the silent "!!!..." spiral: with async scheduling the
        # scheduler only carries -1 spec placeholders; real draft ids live on
        # the last stage. Stage 0 must receive them (upstream: "Broadcast draft
        # proposals so non-last ranks can embed real token ids"), otherwise
        # combine_sampled_and_draft_tokens embeds garbage at every draft
        # position, corrupting KV from the first decode. Upstream #52071 names
        # the failure verbatim: token 0 is embedded -> text fills with '!'.
        (
            """    gen_at_receive_np: np.ndarray  # [num_reqs]""",
            """    gen_at_receive_np: np.ndarray  # [num_reqs]
    # ple-pp-v5 #18: relayed draft proposals [num_reqs, num_speculative_steps]
    draft_tokens: torch.Tensor | None = None""",
        ),
        (
            """        self.max_sample_len = num_speculative_steps + 1
""",
            """        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
""",
        ),
        (
            """    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:""",
            """    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:""",
        ),
        (
            """        self.main_stream.wait_event(slot.event)
        return dict(""",
            """        self.main_stream.wait_event(slot.event)
        # ple-pp-v5 #18: write the relayed draft ids into local req_states.
        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            _dt = slot.draft_tokens
            _dm = slot.idx_mapping
            if exclude_mask.any():
                _keep = torch.as_tensor(~exclude_mask, device=self.device)
                _dt = _dt[_keep]
                _dm = async_copy_to_gpu(
                    slot.idx_mapping_np[~exclude_mask], device=self.device
                )
            draft_tokens_to_update[_dm] = _dt
        return dict(""",
        ),
        (
            """            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()""",
            """            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            # ple-pp-v5 #18: third broadcast carries the draft proposals
            # (order must match the last rank: sampled, combined, drafts).
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            event = self.broadcast_stream.record_event()""",
        ),
        (
            """            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(""",
            """            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(""",
        ),
        (
            """            need_sampled_mask,
            gen_at_receive_np,
        )
        return bool(need_sampled_mask.all())""",
            """            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())""",
        ),
        (
            """            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)""",
            """            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        \"\"\"ple-pp-v5 #18: broadcast draft proposals so non-last ranks can
        embed real token ids (backport of upstream main broadcast_drafts).\"\"\"
        assert self.is_last_rank
        if self.num_speculative_steps == 0:
            return
        if compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            # Must record the idx_mapping tensor since it was allocated
            # on the main stream.
            input_batch.idx_mapping.record_stream(self.broadcast_stream)
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )
            send.record_stream(self.main_stream)""",
        ),
        # ---- 19) Ring-drift tripwire + self-repair (incident: silent acceptance
        # collapse to 0 mid-request, RFC #44697 class: the non-last rank's
        # num_computed_tokens / num_accepted_tokens advance from the RING's
        # broadcast accounting; one cadence flip on a mixed prefill+decode step
        # leaves the GDN rollback and positions permanently off-phase ->
        # deterministic garbage + 0% acceptance, no error anywhere.
        # The GPU counter receives exactly three ops on stage 0: +query_len at
        # receive (postprocess_num_computed_tokens, every row), -num_rejected at
        # ring consumption (post_update with query_start_loc=None, only rows
        # surviving the freed/need-mask filter). Mirror them CPU-side and check
        # against the scheduler's authoritative num_computed_tokens (which the
        # last rank's real sampled counts feed) in update_requests, every step.
        (
            """    # ple-pp-v5 #18: relayed draft proposals [num_reqs, num_speculative_steps]
    draft_tokens: torch.Tensor | None = None""",
            """    # ple-pp-v5 #18: relayed draft proposals [num_reqs, num_speculative_steps]
    draft_tokens: torch.Tensor | None = None
    # ple-pp-v5 #19: pinned CPU mirrors taken on the broadcast stream (no sync)
    num_sampled_np: np.ndarray | None = None
    num_rejected_np: np.ndarray | None = None""",
        ),
        (
            '''        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )''',
            '''        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

        # ple-pp-v5 #19: CPU mirror of the GPU num_computed_tokens counter and
        # per-request generation of that mirror. drift_enabled/repair via env.
        self.drift_enabled = os.environ.get("VLLM_PLE_PP_DRIFT", "0") == "1"
        self.pp_size = get_pp_group().world_size
        self.shadow_np = np.zeros(max_num_reqs, dtype=np.int64)
        self.shadow_gen_np = np.full(max_num_reqs, -1, dtype=np.int32)
        # Pre-allocated pinned mirrors, ring of pp_size buffers: receive at
        # step S writes buf S%pp_size, consumed at step S+pp_size whose
        # update_pp_decode_requests runs BEFORE that step's receive — never
        # overwritten while still referenced.
        self._pinned = None
        if self.drift_enabled and not self.is_last_rank:
            self._pinned = (
                torch.zeros(self.pp_size, max_num_reqs, dtype=torch.int32,
                            pin_memory=True),
                torch.zeros(self.pp_size, max_num_reqs, dtype=torch.int32,
                            pin_memory=True),
            )
        self.drift_count = 0
        self.repair_count = 0
        self.step = 0''',
        ),
        (
            """            event = self.broadcast_stream.record_event()""",
            """            # ple-pp-v5 #19: pinned mirrors for the drift tripwire (ring of
            # pp_size buffers; safe because each step's consume precedes its
            # receive, so the buffer's previous reader is always done).
            num_sampled_np = None
            num_rejected_np = None
            if self._pinned is not None:
                _k = self.step % self.pp_size
                num_sampled_np = self._pinned[0][_k][:num_reqs]
                num_rejected_np = self._pinned[1][_k][:num_reqs]
                num_sampled_np.copy_(combined[0], non_blocking=True)
                num_rejected_np.copy_(combined[1], non_blocking=True)
            event = self.broadcast_stream.record_event()""",
        ),
        (
            """            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())""",
            """            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
            num_sampled_np,
            num_rejected_np,
        )
        return bool(need_sampled_mask.all())""",
        ),
        (
            """        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)""",
            """        assert not self.is_last_rank
        if self.drift_enabled:
            # ple-pp-v5 #19: mirror postprocess_num_computed_tokens (+query_len
            # for EVERY row, even rows that sample nothing), BEFORE the early
            # return — the GPU kernel advances all rows regardless of the mask.
            self.step += 1
            _idx = input_batch.idx_mapping_np
            _q = np.asarray(input_batch.num_scheduled_tokens, dtype=np.int64)
            np.add.at(self.shadow_np, _idx, _q)
            self.shadow_gen_np[_idx] = self.req_idx_gen_np[_idx]
        need_sampled_mask = compute_need_sampled_mask(input_batch)""",
        ),
        (
            """    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1""",
            """    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def on_req_added(self, req_idx: int, num_computed_tokens: int) -> None:
        # ple-pp-v5 #19: seed the shadow counter when a req (re)enters.
        self.shadow_np[req_idx] = num_computed_tokens
        self.shadow_gen_np[req_idx] = self.req_idx_gen_np[req_idx]

    def check_scheduler_drift(
        self,
        computed_gpu: torch.Tensor,
        req_id_to_index: dict,
        req_ids: list,
        sched_num_computed,
        prefill_len_np: np.ndarray,
    ) -> None:
        \"\"\"ple-pp-v5 #19: ring-vs-scheduler tripwire (RFC #44697 class).

        The scheduler's num_computed_tokens is AUTHORITATIVE (fed by the last
        rank's real sampler); shadow_np mirrors ONLY what the PP ring told
        stage 0's GPU counter (+query_len at receive, -num_rejected at ring
        consumption). Legitimate steady state: shadow >= sched by at most the
        ring's lag (rejections the scheduler committed but the ring feeds back
        pp_size steps later) ~= (pp_size+1)*(k+1). The #44697 failure is a
        SKIPPED correction -> shadow - sched grows unboundedly (+k per fully
        rejected step) -> off-by-N positions + mis-rolled-back GDN state ->
        garbage tokens and exactly 0% acceptance with no errors anywhere.
        Trip: shadow - sched beyond 2x the legitimate band (self-healing via
        repair is strictly safer than the silent spiral). Repair: snap GPU +
        shadow to the scheduler value and zero the request's still-pending
        ring slots' num_rejected so stale corrections aren't subtracted twice.
        \"\"\"
        if self.is_last_rank or not self.drift_enabled:
            return
        tol_hi = 2 * (self.pp_size + 1) * (self.num_speculative_steps + 1)
        repair = os.environ.get("VLLM_PLE_PP_DRIFT_REPAIR", "0") == "1"
        for req_id, sched in zip(req_ids, sched_num_computed):
            idx = req_id_to_index.get(req_id)
            if idx is None or self.shadow_gen_np[idx] != self.req_idx_gen_np[idx]:
                continue
            if self.shadow_np[idx] < prefill_len_np[idx]:
                continue
            d = int(self.shadow_np[idx]) - int(sched)
            if int(sched) < int(prefill_len_np[idx]):
                # Scheduler still prefilling this req: cadence doesn't
                # constrain chunked prefill; check only true decode rows.
                continue
            if -8 <= d <= tol_hi:
                continue
            self.drift_count += 1
            if self.drift_count <= 20:
                logger.warning(
                    "[ple-pp #19] DRIFT req=%s idx=%d step=%d ring_shadow=%d "
                    "scheduler=%d delta=%+d tol=%d repair=%s drift_total=%d",
                    req_id, idx, self.step, int(self.shadow_np[idx]), int(sched),
                    d, tol_hi, repair, self.drift_count,
                )
            if repair and d > tol_hi:
                computed_gpu[idx] = int(sched)
                self.shadow_np[idx] = int(sched)
                # Discard this request's still-pending ring corrections so the
                # discarded over-advance is not subtracted again at consume.
                for slot in self.queue:
                    if slot is None or slot.num_rejected is None:
                        continue
                    hit = slot.idx_mapping_np == idx
                    if hit.any():
                        # num_rejected is a view of `combined`, which was
                        # filled on broadcast_stream — sync the recv event
                        # first so this main-stream write can't race it.
                        # (Repair is the rare genuine-drift path.)
                        slot.event.synchronize()
                        slot.num_rejected_np.numpy()[hit] = 0
                        slot.num_rejected[torch.as_tensor(
                            hit, device=slot.num_rejected.device
                        )] = 0
                self.repair_count += 1
        if self.step and self.step % 200 == 0 and (
            self.drift_count or self.repair_count
        ):
            logger.info(
                "[ple-pp #19] stats step=%d drifts=%d repairs=%d",
                self.step, self.drift_count, self.repair_count,
            )""",
        ),
        (
            "@dataclass",
            "logger = init_logger(__name__)\n\n\n@dataclass",
        ),
        (
            "from vllm.distributed.parallel_state import get_pp_group",
            "from vllm.distributed.parallel_state import get_pp_group\nfrom vllm.logger import init_logger",
        ),
        (
            """        self.main_stream.wait_event(slot.event)""",
            """        # ple-pp-v5 #19: mirror post_update's -num_rejected (rows the kernel
        # will actually touch: not excluded). Host read of the pinned mirror is
        # gated by the recv event (microseconds; it was recorded pp_size steps
        # ago, so synchronize() is effectively free).
        if self.drift_enabled and slot.num_rejected_np is not None:
            slot.event.synchronize()
            _keep = ~exclude_mask
            _rows = slot.idx_mapping_np[_keep]
            if _rows.size:
                np.add.at(
                    self.shadow_np, _rows,
                    -slot.num_rejected_np.numpy()[_keep].astype(np.int64),
                )
        self.main_stream.wait_event(slot.event)""",
        ),
    ],
)

# ---------------------------------------------------------------- 18b) MRV2 call sites
MRV2 = f"{PKG}/v1/worker/gpu/model_runner.py"
edit(
    MRV2,
    [
        # Non-last rank: hand req_states.draft_tokens to the relay consumer.
        (
            """            outputs = self.pp_handler.get_prev_sampled_outputs()""",
            """            outputs = self.pp_handler.get_prev_sampled_outputs(
                self.req_states.draft_tokens
            )""",
        ),
        # ple-pp-v5 #19: seed the drift shadow when a req enters (or is
        # re-added by streaming input), from the scheduler's authoritative
        # starting count. Runs BEFORE the same step's receive() mirror.
        (
            """            req_index = self.req_states.req_id_to_index[req_id]
            if self.adaptive_verification is not None:""",
            """            req_index = self.req_states.req_id_to_index[req_id]
            # ple-pp-v5 #19: seed ring-drift shadow from authoritative count.
            if self.pp_handler is not None:
                self.pp_handler.on_req_added(
                    req_index, new_req_data.num_computed_tokens
                )
            if self.adaptive_verification is not None:""",
        ),
        # ple-pp-v5 #19: run the ring-vs-scheduler tripwire once per step, at
        # the end of update_requests (scheduler values just landed, ring
        # consumption for this step already done in update_pp_decode_requests).
        (
            """                    req_index, req_new_block_ids, overwrite=False
                )

        # Update CPU num_computed_prefill_tokens.""",
            """                    req_index, req_new_block_ids, overwrite=False
                )

        # ple-pp-v5 #19: ring-vs-scheduler drift tripwire (no-op on last rank
        # and unless VLLM_PLE_PP_DRIFT=1).
        if self.pp_handler is not None and reqs.req_ids:
            self.pp_handler.check_scheduler_drift(
                self.req_states.num_computed_tokens.gpu,
                self.req_states.req_id_to_index,
                list(reqs.req_ids),
                reqs.num_computed_tokens,
                self.req_states.prefill_len.np,
            )

        # Update CPU num_computed_prefill_tokens.""",
        ),
        # Last rank: relay the freshly proposed drafts to earlier stages
        # (same position as upstream, inside the num_speculative_steps>0 guard).
        (
            """            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )""",
            """            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )
            if self.pp_handler is not None:
                # ple-pp-v5 #18: relay real draft ids to earlier stages.
                self.pp_handler.broadcast_drafts(
                    self.req_states.draft_tokens, input_batch
                )""",
        ),
    ],
)

# ---------------------------------------------------------------- 8) #20
# VLLM_PP_LAYER_PARTITION is process-global. The PLE offload worker is a
# standalone CPU process (gloo world=1, "isolated rank-zero, world-size-one
# group") that rebuilds the FULL 48-layer meta model for weight discovery -
# it is NOT a pipeline stage, so the 2-element partition string against its
# pp_size=1 crashed startup with
#   ValueError: len(partitions)=2 does not match pp_size=1
# propagated through wait_ple_offload_ready(). Mark the process and make
# get_pp_indices ignore the partition list there: a uniform split of
# pp_size=1 hands it all layers, exactly the v4 behavior.
PLEW = f"{PKG}/v1/ple_offload/worker.py"
edit(
    PLEW,
    [
        (
            """            _init_offload_distributed()""",
            """            # ple-pp-v5 #20: not a pipeline stage; the serving workers
            # export VLLM_PP_LAYER_PARTITION to every process in the
            # container. get_pp_indices must ignore it in this one.
            os.environ["VLLM_PLE_OFFLOAD_WORKER"] = "1"
            _init_offload_distributed()""",
        ),
    ],
)

DISTUTILS = f"{PKG}/distributed/utils.py"
edit(
    DISTUTILS,
    [
        (
            """    partition_list_str = envs.VLLM_PP_LAYER_PARTITION
    if partition_list_str is not None:""",
            """    partition_list_str = envs.VLLM_PP_LAYER_PARTITION
    if (
        partition_list_str is not None
        and os.environ.get("VLLM_PLE_OFFLOAD_WORKER") == "1"
    ):
        # ple-pp-v5 #20: see v1/ple_offload/worker.py - the offload worker
        # needs every layer (uniform split of pp_size=1), never the stage
        # partition list armed for the serving workers.
        partition_list_str = None
    if partition_list_str is not None:""",
        ),
    ],
)

print("[ple-pp] all patches applied")
sys.exit(0)
