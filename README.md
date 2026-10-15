# ple7-vm-pack

One-script deploy of the **ple7 v7** stack — vLLM fork + PLE CPU/SSD offload +
PP2 + MTP k=4 + drift tripwire — serving **Qwen3.8-Flash-Next NVFP4** on a
VMware VM with GPU passthrough.

On a fresh Ubuntu 22.04/24.04 VM (GPU passed through, ≥96 GB RAM, ≥400 GB
free disk):

```
curl -fsSL https://raw.githubusercontent.com/M4G4MED/ple7-vm-pack/main/ple7-deploy.sh -o ple7-deploy.sh
sudo bash ple7-deploy.sh
```

(Or paste the script onto the machine and run `sudo bash ple7-deploy.sh`.)

It installs, in order: NVIDIA open kernel driver (apt) → podman (apt) →
the vLLM image (`build/Dockerfile` chains the public
`vllm/vllm-openai:qwen38-flash-next` base with the 7 patch scripts in this
repo, reproducing production image `ple-offload:v7` bit-for-bit in behavior) →
model weights from `huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4` (plain
HTTPS, resume-safe, **no HF token**) → the 48 GB PLE n-gram table, extracted
locally from the downloaded checkpoint and certified against it with a
fingerprint + 64-row byte sample — **no separate table download** → a systemd
daemon `ple7-vm.service` that starts the server and restarts it on failure.

Re-runnable: every step skips or resumes when already done (the driver step
exits 90 once and asks for a reboot).

## Requirements (honest list)
- Full **PCI passthrough** of the GPU to the guest. VMware vGPU profiles do
  not work (no full device access / wrong VRAM accounting).
- **Driver ≥ 580.65** (open kernel module). The image is CUDA 13.0; a 570
  driver installs fine but CUDA init fails at first token. The script checks
  this and refuses to proceed with an old driver.
- **Secure Boot off** (or a MOK-enrolled DKMS module) — Ubuntu's `nvidia-open`
  is DKMS-built and won't load under Secure Boot. The script detects this.
- **ESXi only:** BIOS "Above 4G Decoding" on, VM firmware **EFI**, memory
  **fully reserved**, and these VM advanced settings (big GPUs have huge
  BARs — without them the VM won't power on or the driver won't bind):
  `pciPassthru.use64bitMMIO = TRUE` and `pciPassthru.64bitMMIOSizeGB` = total
  GPU VRAM summed, rounded up to the next power of two, **then one power
  higher** (2×64 GB → 256; 2×48 GB → 128). Resizable-BAR is unsupported in
  passthrough mode — leave it off on the host.
- **Workstation only:** no ECC/vGPU issues, but passthrough requires the host
  GPU to be the non-primary device or using `cuda.enable = TRUE` unsupported
  configs — realistically ESXi (or KVM) is the right hypervisor for this.
- **sm_80 or newer**, ≥48 GiB per GPU for PP=2 (production: 2× 64 GB), or a
  single ≥76 GiB card with `PP_SIZE=1`.
- ≥96 GB VM RAM (the 48 GB PLE table is mmap'd), ≥400 GB free disk.
- The image runs Ampere kernels (that's what the production cards are) —
  NVFP4 weights dequantize through the fork's sm_80 path.
- Docker Hub anonymous pulls are rate-limited per IP; on shared NAT IPs
  `podman pull` of the 8.6 GB base may need a `podman login docker.io` retry.

## Tuning
Edit `/etc/systemd/system/ple7-vm.service` or `/usr/local/bin/ple7-vm`
headers: `PORT GPU_ORDER MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS SPEC_K
ENABLE_MTP ASYNC DRIFT_MONITOR PP_SIZE`. `MAX_NUM_BATCHED_TOKENS` is capped
at 8192 on purpose (GDN prefill OOMs above it — production-proven).

## Layout
- `ple7-deploy.sh` — the installer (paste target)
- `ple7-vm.sh` — launcher (fetched to `/usr/local/bin/ple7-vm`; commands:
  `start|stop|status|logs|daemon`)
- `build/Dockerfile` + `build/patch_*.py` + `build/ple_ssd.py` — the exact
  patch chain applied over the public vLLM image, layer by layer (v1 gate +
  MTP-quant + blockfp8 + SSD-offload → v5 PP → v6 agentic stability → v7
  cross-request state backports).
