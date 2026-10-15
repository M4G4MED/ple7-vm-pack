# Rebuild of ple-offload:v7 from public base — single Containerfile, exact layer order
FROM docker.io/vllm/vllm-openai@sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3 AS base
# was: :qwen38-flash-next — digest pinned 2026-10-15 (mutable tags can drift;
# this exact manifest was diffed byte-identical against production ple-offload:v7)
COPY patch_ple_gate.py /tmp/patch_ple_gate.py
RUN python3 /tmp/patch_ple_gate.py && rm /tmp/patch_ple_gate.py
COPY patch_mtp_quant.py /tmp/patch_mtp_quant.py
RUN python3 /tmp/patch_mtp_quant.py && rm /tmp/patch_mtp_quant.py
COPY patch_modelopt_blockfp8.py /tmp/patch_modelopt_blockfp8.py
RUN python3 /tmp/patch_modelopt_blockfp8.py && rm /tmp/patch_modelopt_blockfp8.py
COPY ple_ssd.py /usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/ple_ssd.py
COPY patch_ple_ssd.py /tmp/patch_ple_ssd.py
RUN python3 /tmp/patch_ple_ssd.py && rm /tmp/patch_ple_ssd.py
COPY patch_ple_pp.py /tmp/patch_ple_pp.py
RUN python3 /tmp/patch_ple_pp.py && rm /tmp/patch_ple_pp.py
COPY patch_ple_pp_v6.py /tmp/patch_ple_pp_v6.py
RUN python3 /tmp/patch_ple_pp_v6.py && rm /tmp/patch_ple_pp_v6.py
COPY patch_ple_pp_v7.py /tmp/patch_ple_pp_v7.py
RUN python3 /tmp/patch_ple_pp_v7.py && rm /tmp/patch_ple_pp_v7.py
