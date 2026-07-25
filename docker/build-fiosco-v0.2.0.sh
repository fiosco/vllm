#!/usr/bin/env bash
# Build the Fiosco vLLM v0.2.0 overlay image.
#
# Two target platforms, one per GPU family:
#   amd64 -> fiosco/vllm:v0.2.0-amd64  for SM120  (RTX 5000/6000 PRO Blackwell, cc 12.0)
#   arm64 -> fiosco/vllm:v0.2.0-arm64  for sm_121 (DGX Spark GB10, cc 12.1, aarch64)
#
# Build each arch on a NATIVE node (amd64 on an x86_64 host / rtxforge; arm64 on
# a DGX Spark) to avoid qemu cross-emulation. The official base is multi-arch, so
# --platform selects the matching manifest.
#
#   ./docker/build-fiosco-v0.2.0.sh          # host's native arch
#   ./docker/build-fiosco-v0.2.0.sh amd64    # force amd64 (needs amd64 host or buildx+qemu)
#   ./docker/build-fiosco-v0.2.0.sh arm64    # force arm64 (needs arm64 host or buildx+qemu)
#
# Env overrides: BASE=<base image>  TAG=<full tag>
set -euo pipefail

BASE=${BASE:-vllm/vllm-openai:v0.26.0}
ARCH=${1:-$(uname -m)}
case "$ARCH" in
  x86_64|amd64)  PLAT=linux/amd64;  SUF=amd64 ;;
  aarch64|arm64) PLAT=linux/arm64;  SUF=arm64 ;;
  *) echo "unknown arch: $ARCH (use amd64 | arm64)"; exit 1 ;;
esac
TAG=${TAG:-fiosco/vllm:v0.2.0-$SUF}

cd "$(git rev-parse --show-toplevel)"
echo ">> building $TAG  (platform=$PLAT, base=$BASE)"
docker build --platform "$PLAT" --build-arg BASE="$BASE" \
  -f docker/Dockerfile.fiosco-overlay -t "$TAG" .
echo ">> done: $TAG"
echo
echo "   SM120  (RTX Blackwell)  : fiosco/vllm:v0.2.0-amd64  (DeepGEMM sm121 alias is a no-op here)"
echo "   sm_121 (DGX Spark GB10) : fiosco/vllm:v0.2.0-arm64  (build on a Spark; DeepGEMM sm121 alias active)"
