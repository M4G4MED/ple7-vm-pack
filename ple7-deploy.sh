#!/usr/bin/env bash
# =============================================================================
# ple7-vm deploy — Qwen3.8-Flash-Next NVFP4 (vLLM + PLE offload + PP + MTP)
# Paste-on-VM installer. Needs: Ubuntu 22.04/24.04 VM, NVIDIA GPU(s) passed
# through to VMware (full PCI passthrough, not vGPU profiles), >=96 GB RAM
# (128 recommended), >=400 GB free disk on BASE.
#
#   sudo bash ple7-deploy.sh
#
# Everything comes from the public web: NVIDIA driver (apt), podman+git (apt),
# vLLM base image (hub.docker.com/r/vllm/vllm-openai:qwen38-flash-next) +
# our patches (github.com/M4G4MED/ple7-vm-pack), model weights
# (huggingface.co). The 48 GB PLE table is derived locally from the model
# file — bit-identical to the production table, no extra download.
# No credentials needed anywhere. Re-runnable: steps resume/skip when done.
# =============================================================================
set -uo pipefail

BASE=${BASE:-/opt/ple7}
MODEL_DIR=${MODEL_DIR:-$BASE/model}
TABLE_DIR=${TABLE_DIR:-$BASE/ple-ssd}
CACHE_DIR=${CACHE_DIR:-$BASE/cache}
IMAGE=${IMAGE:-localhost/ple-offload:v7}
REPO=nvidia/Qwen3.8-Flash-Next-NVFP4
GH=https://github.com/M4G4MED/ple7-vm-pack
PORT=${PORT:-8080}
log() { echo "[ple7-deploy $(date +%H:%M:%S)] $*"; }
die() { log "FATAL: $*"; exit 1; }

[ "$(id -u)" = 0 ] || die "run as root: sudo bash ple7-deploy.sh"
grep -qi ubuntu /etc/os-release || log "WARN: not Ubuntu — steps 1-2 assume apt/dpkg; continue only if you know why"
mkdir -p "$MODEL_DIR" "$TABLE_DIR" "$CACHE_DIR"

# ---------- [1/6] NVIDIA open kernel driver ----------
if ! nvidia-smi >/dev/null 2>&1; then
  log "[1/6] installing NVIDIA open kernel module…"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -yq software-properties-common
  add-apt-repository -y universe || true
  apt-get install -yq nvidia-open \
    || apt-get install -yq nvidia-driver-570-server-open \
    || apt-get install -yq nvidia-driver-570-open \
    || die "no suitable nvidia driver package — install an open-module driver >=565 manually, then re-run"
  log "[1/6] driver installed. REBOOT, then: sudo bash ple7-deploy.sh"
  exit 90
fi
NG=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NG" -ge 1 ] || die "driver present but 0 GPUs visible — check VMware PCI passthrough"
log "[1/6] driver OK: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | paste -sd' | ')"

# ---------- [2/6] podman + git ----------
if ! command -v podman >/dev/null 2>&1; then
  log "[2/6] installing podman git curl…"
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -yq podman git curl ca-certificates
else
  log "[2/6] podman OK"
fi

# ---------- [3/6] build the vLLM image from public chain ----------
if ! podman image inspect "$IMAGE" >/dev/null 2>&1; then
  log "[3/6] building $IMAGE: pulls public vllm base (~9 GB compressed) + applies our 7 patches (~15 min)…"
  rm -rf "$BASE/repo"
  git clone --depth 1 "$GH.git" "$BASE/repo" || die "cannot clone $GH"
  podman build -t "$IMAGE" -f "$BASE/repo/build/Dockerfile" "$BASE/repo/build" \
    || die "image build failed — check network to hub.docker.com (anonymous pull rate limits apply)"
else
  log "[3/6] image present"
fi

# ---------- [4/6] model weights (plain HTTPS, resume-safe, no token) ----------
IDX="$MODEL_DIR/.dl-index.json"
[ -s "$IDX" ] || curl -fsSL --retry 5 --retry-delay 3 \
  "https://huggingface.co/api/models/$REPO/tree/main?recursive=true" -o "$IDX" \
  || die "cannot list $REPO from huggingface.co"
need=0
while read -r f sz; do
  case "$f" in *.safetensors|*.json|*.model|*.txt|*.py) ;; *) continue ;; esac
  out="$MODEL_DIR/$f"
  if [ ! -f "$out" ] || [ "$(stat -c%s "$out" 2>/dev/null)" != "$sz" ]; then
    need=1
    log "[4/6] downloading $f ($sz bytes)…"
    for a in 1 2 3; do
      curl -fL --retry 5 --retry-delay 5 -C - -o "$out" \
        "https://huggingface.co/$REPO/resolve/main/$f" && break
      log "  retry $a for $f"
    done
    [ "$(stat -c%s "$out" 2>/dev/null)" = "$sz" ] || die "$f incomplete after retries"
  fi
done < <(python3 - "$IDX" <<'PY'
import json,sys
for e in json.load(open(sys.argv[1])):
    if e.get("type")=="file": print(e["path"], e.get("size",0))
PY
)
[ "$need" = 0 ] && log "[4/6] model already complete at $MODEL_DIR"
SRC="$MODEL_DIR/model-fp8-mtp-ple.safetensors"
[ -f "$SRC" ] || die "model-fp8-mtp-ple.safetensors missing after download"

# ---------- [5/6] PLE n-gram table (derived from the checkpoint, verified) ----------
TBL="$TABLE_DIR/ple_ngram.table"
if [ -f "$TBL" ] && [ -f "$TBL.meta" ]; then
  log "[5/6] table present (runtime will byte-verify it at start)"
else
  log "[5/6] extracting PLE table from the checkpoint (~51 GB copy, minutes; certifies itself against the source)…"
  python3 - "$SRC" "$TBL" <<'PY'
import json,os,struct,sys,hashlib,time
src,tbl=sys.argv[1],sys.argv[2]
with open(src,"rb") as f:
    hl=struct.unpack("<Q",f.read(8))[0]
    hbytes=f.read(hl); ds=8+hl
    hdr=json.loads(hbytes)
    sh={k:(m["data_offsets"][0],m["data_offsets"][1]) for k,m in hdr.items()
        if ".ngram_embedding." in k and ".shard_" in k and k.endswith(".weight")
        and m.get("dtype")=="F8_E4M3" and len(m.get("shape",[]))==2}
    assert sh, "ngram shards not found in checkpoint"
    span=[min(v[0] for v in sh.values()), max(v[1] for v in sh.values())]
    n=span[1]-span[0]
    f.seek(ds+span[0])
    tmp=tbl+".part"
    with open(tmp,"wb") as o:
        r=0
        while r<n:
            b=f.read(min(1<<24,n-r))
            if not b: break
            o.write(b); r+=len(b)
        o.flush(); os.fsync(o.fileno())
    assert r==n, f"short copy {r}!={n}"
    # certify a sample against the source
    import random
    random.seed(1234)
    with open(tbl+".part","rb") as t:
        for _ in range(64):
            off=random.randrange(n-65536)
            f.seek(ds+span[0]+off); t.seek(off)
            assert f.read(65536)==t.read(65536), f"sample mismatch at {off}"
    digest=hashlib.sha256(hbytes).hexdigest()
    json.dump({"digest":digest,"source_size":os.path.getsize(src),"bytes":n,
               "rows":n//160,"cols":160,
               "built_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},
              open(tbl+".meta.tmp","w"))
    os.replace(tmp,tbl); os.replace(tbl+".meta.tmp",tbl+".meta")
    print(f"table OK: {n} bytes, 64/64 samples match source, digest {digest[:16]}..", file=sys.stderr)
PY
  [ -f "$TBL" ] || die "table extraction failed"
fi

# ---------- [6/6] launcher + systemd ----------
curl -fsSL --retry 5 -o /usr/local/bin/ple7-vm \
  "$GH/raw/main/ple7-vm.sh" || die "cannot fetch ple7-vm.sh from $GH"
chmod 0755 /usr/local/bin/ple7-vm
cat >/etc/systemd/system/ple7-vm.service <<'UNIT'
[Unit]
Description=vLLM Qwen3.8-Flash-Next NVFP4 (ple7 v7)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=MODEL=/opt/ple7/model
Environment=TABLE_DIR=/opt/ple7/ple-ssd
Environment=CACHE_DIR=/opt/ple7/cache
ExecStart=/usr/local/bin/ple7-vm daemon
Restart=on-failure
RestartSec=15
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable ple7-vm.service
log "ALL DONE — starting server…"
systemctl start ple7-vm
cat <<EOF

  watch :  journalctl -u ple7-vm -f     (wait for 'Application startup complete')
  api   :  http://127.0.0.1:$PORT/v1    model name: qwen3.8-flash-next
  test  :  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\
           -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
EOF
