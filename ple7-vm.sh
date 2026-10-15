#!/usr/bin/env bash
# =============================================================================
# ple7-vm.sh — Qwen3.8-Flash-Next NVFP4 (PLE-offload + PP + MTP) for a VMware
# VM with NVIDIA GPU passthrough. Port of the production /home/server/ple7.sh
# v7 stack, with the CMP-170HX-specific preflight replaced by generic GPU
# discovery. Everything else (image, args, patches) is byte-for-byte the
# production command line.
#
# Layout on the VM (created by setup-vm.sh):
#   model    : /opt/ple7/model          (nvidia/Qwen3.8-Flash-Next-NVFP4)
#   table    : /opt/ple7/ple-ssd/ple_ngram.table (+ .meta)
#   caches   : /opt/ple7/cache
#   log      : /var/log/ple7.log
#
# Usage: ple7-vm.sh start|stop|status|logs|daemon
#   daemon = ensure container + podman start -a in FOREGROUND (systemd ExecStart)
# =============================================================================
set -u
set -o pipefail

BASE=${PLE7_BASE:-/opt/ple7}
MODEL=${MODEL:-$BASE/model}
TABLE_DIR=${TABLE_DIR:-$BASE/ple-ssd}
CACHE_DIR=${CACHE_DIR:-$BASE/cache}
LOGFILE=${LOGFILE:-/var/log/ple7.log}
PIDFILE=/run/ple7-vm.pid
IMAGE=${IMAGE:-localhost/ple-offload:v7}
CONTAINER=${CONTAINER:-vllm-ple7}
PORT=${PORT:-8080}
SERVED_NAME=${SERVED_NAME:-qwen3.8-flash-next}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}   # NEVER raise: GDN prefill scratch OOMs above
MAX_NUM_SEQS=${MAX_NUM_SEQS:-9}
SPEC_K=${SPEC_K:-4}
ENABLE_MTP=${ENABLE_MTP:-1}
ASYNC=${ASYNC:-1}
DRIFT_MONITOR=${DRIFT_MONITOR:-1}
GPU_ORDER=${GPU_ORDER:-}                          # e.g. "1,0"; empty = natural order

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGFILE" 2>/dev/null || echo "$*"; }

PODMAN=$(command -v podman || true)
[ -n "$PODMAN" ] || { echo "podman missing — run setup-vm.sh"; exit 1; }

ngpu() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n; }

# MAX_MODEL_LEN heuristic: 196608 needs the big HBM cards; scale down with VRAM.
default_maxlen() {
  local min_mib; min_mib=$(ngpu | head -1); min_mib=${min_mib:-0}
  if   [ "$min_mib" -ge 63000 ]; then echo 196608
  elif [ "$min_mib" -ge 46000 ]; then echo 131072
  elif [ "$min_mib" -ge 40000 ]; then echo 65536
  else echo 32768; fi
}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(default_maxlen)}

TP_SIZE=${TP_SIZE:-1}
# topology: 2+ GPUs -> PP2; 1 GPU -> PP1 (only sane if that one GPU is huge)
if [ "$(ngpu | wc -l)" -ge 2 ]; then PP_SIZE=${PP_SIZE:-2}; else PP_SIZE=${PP_SIZE:-1}; fi

resolve_drv() {
  for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do
    [ -e "$d/libnvidia-ml.so.1" ] && { echo "$d"; return 0; }
  done
  return 1
}

preflight() {
  local fail=0
  $PODMAN image inspect "$IMAGE" >/dev/null 2>&1 \
    || { log "  FAIL: image $IMAGE absent (podman load -i image-vllm-ple7-v7.tar)"; fail=1; }
  [ -f "$MODEL/config.json" ] \
    || { log "  FAIL: model not downloaded at $MODEL (setup-vm.sh step 3)"; fail=1; }
  local min_vram; min_vram=$(ngpu | head -1)
  if [ -z "$min_vram" ]; then log "  FAIL: nvidia-smi found nothing — driver not installed?"; fail=1; fi
  local n; n=$(ngpu | wc -l)
  local need=$((TP_SIZE*PP_SIZE))
  [ "$n" -ge "$need" ] || { log "  FAIL: need $need GPUs, VM has $n (check GPU passthrough)"; fail=1; }
  if [ "$PP_SIZE" = "1" ] && [ "${min_vram:-0}" -lt 76000 ]; then
    log "  FAIL: single-GPU PP1 needs >=76 GiB on one GPU (NVFP4 weights ~68 GiB after PLE offload)"; fail=1
  fi
  [ -f "$TABLE_DIR/ple_ngram.table" ] \
    || log "  WARN: PLE SSD table absent — start may take much longer / more RAM (see README)"
  DRV_LIB=$(resolve_drv) || { log "  FAIL: no NVIDIA userspace driver libs found"; fail=1; }
  if [ "$fail" = 0 ]; then log "preflight OK: GPUs=$n minVRAM=${min_vram}MiB PP=$PP_SIZE maxlen=$MAX_MODEL_LEN drv=$DRV_LIB"; fi
  return $fail
}

ctr_args() {
  ADD_ENV=()
  [ -n "$GPU_ORDER" ] && ADD_ENV+=( -e CUDA_VISIBLE_DEVICES="$GPU_ORDER" )
  ARGS=( --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools )
  local i; for i in $(seq 0 $(( TP_SIZE*PP_SIZE - 1 ))); do ARGS+=( --device "/dev/nvidia$i" ); done
  ARGS+=( --device /dev/nvidiactl
    -v "$DRV_LIB":/hostdrv:ro
    -v "$MODEL":/model:ro
    -v "$CACHE_DIR":/cache
    -v "$TABLE_DIR":/ple-ssd
    -e LD_LIBRARY_PATH=/hostdrv:/usr/local/cuda/lib64
    -e VLLM_PLE_CPU_OFFLOAD=1
    -e VLLM_PLE_SSD_OFFLOAD=1
    -e VLLM_PLE_SSD_TABLE_PATH=/ple-ssd/ple_ngram.table
    -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=600
    -e VLLM_PLE_PP_EXPERIMENTAL=1
    -e VLLM_PLE_PP_DRIFT=$DRIFT_MONITOR
    -e VLLM_PLE_PP_DRIFT_REPAIR=$DRIFT_MONITOR
    -e NCCL_P2P_DISABLE=1
    -e TRITON_CACHE_DIR=/cache/triton
    -e TORCH_HOME=/cache/torch
    -e VLLM_LOGGING_LEVEL=INFO
    "${ADD_ENV[@]}"
    --net host --ipc=host
    --name "$CONTAINER"
    --entrypoint vllm
    "$IMAGE"
    serve /model
    --port "$PORT"
    --served-model-name "$SERVED_NAME"
    --tensor-parallel-size "$TP_SIZE"
    --pipeline-parallel-size "$PP_SIZE"
    --enable-prefix-caching
    --prefix-caching-hash-algo sha256
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --reasoning-parser qwen3
    --default-chat-template-kwargs '{"reasoning_effort":"xhigh","enable_thinking":true}'
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --limit-mm-per-prompt '{"image": 8, "video": 1}'
    --media-io-kwargs '{"video":{"fps":6,"num_frames":64}}'
    --mm-processor-kwargs '{"min_pixels":200704,"max_pixels":4194304}'
    --kv-cache-dtype auto
    --block-size 128
    --mamba-ssm-cache-dtype auto
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
  )
  if [ "$ASYNC" = "1" ]; then ARGS+=( --async-scheduling ); else ARGS+=( --no-async-scheduling ); fi
  if [ "$ENABLE_MTP" = "1" ]; then
    ARGS+=( --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":\"$SPEC_K\" }" )
  fi
}

create_ctr() {
  preflight || { log "preflight failed"; return 1; }
  mkdir -p "$CACHE_DIR"
  if $PODMAN container inspect "$CONTAINER" >/dev/null 2>&1; then
    log "container $CONTAINER exists — reusing"
    return 0
  fi
  ctr_args
  log "creating $CONTAINER (PP=$PP_SIZE SPEC_K=$SPEC_K maxlen=$MAX_MODEL_LEN)"
  $PODMAN create "${ARGS[@]}" >/dev/null || { log "FAIL: podman create"; return 1; }
}

get_pid() { $PODMAN container inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; }

case "${1:-status}" in
  start)
    if get_pid; then log "already running"; exit 1; fi
    create_ctr || exit 1
    $PODMAN start "$CONTAINER" >/dev/null
    log "started. logs: $0 logs ; health: curl -fsS http://127.0.0.1:$PORT/health"
    ;;
  stop)
    $PODMAN stop -t 60 "$CONTAINER" >/dev/null 2>&1 && log "stopped" || log "was not running"
    ;;
  restart) "$0" stop; sleep 2; "$0" start ;;
  logs) exec $PODMAN logs --tail 200 -f "$CONTAINER" ;;
  status)
    if get_pid; then
      echo "RUNNING"; curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 300; echo
    else echo "STOPPED"; fi
    ;;
  daemon)
    # systemd ExecStart: ensure the container object, then start -a (blocks).
    create_ctr || exit 1
    exec $PODMAN start -a "$CONTAINER"
    ;;
  *) echo "usage: $0 start|stop|restart|status|logs|daemon"; exit 2 ;;
esac
