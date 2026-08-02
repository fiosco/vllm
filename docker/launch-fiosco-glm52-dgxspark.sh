#!/usr/bin/env bash
# launch-fiosco-glm52-dgxspark.sh — production launcher for GLM-5.2 QuantTrio (unpruned) on the
# 4-node DGX Spark cluster (GB10 / sm_121 / aarch64, dual-rail 200G RoCE), vLLM v0.26.0.
#
# Image: fiosco/vllm:v0.2.0-arm64-nccl2.29.7, built from docker/Dockerfile.fiosco-overlay on
# release/cu130-v0.2.0. That image ships NCCL 2.29.7 natively, so there is NO LD_PRELOAD and NO
# bind-mount here: exactly one libnccl build is resolvable and there is no symbol-resolution race.
# Do not add either back. Background: docs/nccl-2289-allgather-hang-findings.md equivalent in
# /mnt/data/tmp/glm52-dcp-serve/ and the commit message of the 2.29.7 pin.
#
# Every parameter below is transcribed from the deployment that is serving production, verified
# against `docker inspect vllm_v026_256k` on dgxspark1 (Config.Cmd, Config.Env, HostConfig.Binds).
# Older launch scripts in /mnt/data/tmp/ carry retired settings — do not derive from them.
#
# ==============================================================================================
# EVIDENCE STATUS — read before selecting a mode. The three modes are NOT equally supported.
# ==============================================================================================
#
# 256k (default) — PROVEN. Validated end to end on 2026-07-30 on this cluster, this image, this
#   exact command line:
#     36/36 requests HTTP 200 over 29m12s (1752 s), zero errors, zero retries
#     23.08 tok/s aggregate on the 700-token generation series (4 runs, 2800 tok / 121.323 s)
#     50.05% draft acceptance on that series (1868/3732)
#     nine ~109,754-token long-context requests spanning only 1.43 s (146.27–147.70 s) across
#       25 minutes — no drift
#     RAM 97.6–98.7% used, swap flat to -13 Mi, no monotonic degradation
#     clocks_event_reasons.active 0x0 at all five samples on all four nodes — no ACTIVE thermal
#       throttling (lifetime counters are non-zero but were not snapshotted before the window,
#       so nothing can be attributed to it)
#     ERROR 0, CRITICAL 0, Traceback 0, "out of memory" 0, NCCL WARN 0 across all four nodes
#
# 288k and 320k — UNVALIDATED ON v0.26.0. Neither has ever been started on this image. Their
#   byte counts are derived arithmetically from the 256k measurement (3,504,012 B/block, measured
#   from the 256k run's own KV sizing log) and nothing else. There is no throughput, latency,
#   acceptance, or stability data for either.
#
#   The memory arithmetic, from the 256k run's idle sample (T0, all four nodes idle and serving):
#     measured MemAvailable   dgxspark1 1887 MiB   dgxspark2 3021   dgxspark3 2985   dgxspark4 3018
#     288k costs +0.88 GiB/node over 256k  -> dgxspark1 left with roughly  985 MiB
#     320k costs +1.76 GiB/node over 256k  -> dgxspark1 left with roughly   89 MiB
#   dgxspark1 is the head and runs EngineCore, which is why it is ~1.1 GiB tighter than the
#   workers; it is the node that binds.
#
#   Say plainly what 320k means: 89 MiB of headroom on the head node RE-CREATES THE MEMORY
#   PRESSURE THE MOVE TO 256k WAS MADE TO ESCAPE. The prior deployment sat at 1098 MiB available
#   on dgxspark1 with 3327 MiB of swap in use, and that is the condition 256k was chosen to get
#   out of. Do not run 320k against production. If either larger mode is wanted, run the same
#   validation the 256k figures above came from, on a window the cluster can afford to lose.
#
# ==============================================================================================
# KV SIZING — --kv-cache-memory-bytes, NOT --num-gpu-blocks-override
# ==============================================================================================
#
# --num-gpu-blocks-override IS RETIRED. Do not reintroduce it.
#   --gpu-memory-utilization is a TOTAL-footprint envelope and KV is the RESIDUAL, so per-rank
#   variance in weights / activations / non-torch memory lands entirely on KV. Profiling the four
#   ranks produced 8.29 / 7.42 / 6.56 / 6.95 GiB of "available KV" = natural block counts of
#   2540 / 2274 / 2010 / 2131. An override of 2150 against those counts is not a cap: vLLM
#   overrides UPWARD too, and it forced two ranks up (dgxspark3 by 140 blocks, dgxspark4 by 19),
#   over-committing the two tightest workers.
#   --kv-cache-memory-bytes is per-GPU, skips memory profiling entirely, and ignores
#   gpu_memory_utilization for sizing. On the validated run all four ranks reserved an identical
#   7.02 GiB and emitted ZERO "Overriding num_gpu_blocks" lines.
#
# THE MAX_SAFE_BLOCKS GUARD FROM EARLIER SCRIPTS IS RETIRED AND MUST NOT BE REINTRODUCED.
#   It existed only to keep --num-gpu-blocks-override below the measured minimum natural
#   allocation, i.e. to defend against the profiling lottery. --kv-cache-memory-bytes eliminates
#   that lottery by skipping profiling, so the guard now defends against nothing, and carrying it
#   would imply a block override is still in play. There is no block override in this script.
#
# --gpu-memory-utilization 0.90 IS RETAINED even though it no longer sizes KV.
#   v1/worker/utils.py:request_memory still raises at init_device when
#   free_memory < total_memory * 0.90, so it gates startup. On the validated run initial free
#   memory was 111.55 / 112.43 / 112.26 / 112.17 GiB and the gate passed on all four ranks.
#
# VLLM_ENABLE_STARTUP_PLAN=1 / startup_plan.py was CONSIDERED AND REJECTED as the sizing
# mechanism: its cache fingerprint includes worker.rank, so it would persist the UNEQUAL per-rank
# values that are the problem, one per rank, rather than converging them.
#
# ==============================================================================================
# MODE TABLE — arithmetic, all three verified against 3,504,012 B/block and 128 tok/block
# ==============================================================================================
#
#   mode   --max-model-len   --kv-cache-memory-bytes   blocks   tokens    concurrency
#   256k          262144                7533625800      2150   275,200      1.0498x
#   288k          294912                8479709040      2420   309,760      1.0503x
#   320k          327680                9418784256      2688   344,064      1.0500x
#
#   128 tok/block is block_size(64) x dcp_size(2). DCP2 is what doubles it. If DCP is ever turned
#   off, one block backs 64 tokens and every byte count above is wrong by 2x. This script always
#   runs DCP2, as production does.
#
# ==============================================================================================
# OPERATIONAL
# ==============================================================================================
#
#   ./launch-fiosco-glm52-dgxspark.sh --dry-run             # print the docker run lines, launch nothing
#   ./launch-fiosco-glm52-dgxspark.sh                       # 256k, the proven configuration
#   ./launch-fiosco-glm52-dgxspark.sh --mode 288k           # UNVALIDATED, requires --i-accept-unvalidated
#   ./launch-fiosco-glm52-dgxspark.sh --stop                # graceful SIGTERM stop, 150 s grace
#
# NEVER SIGKILL / docker kill / docker rm -f on GB10: GPU allocations leak until a cold reboot.
# The stop path is `docker stop -t 150` and nothing else.
#
# Verify after launch (expect ZERO matches on the first one — there is no profiled value left to
# override, and a match means a block override has crept back in):
#   ssh -4 dgxspark1 "docker logs vllm_v026_256k 2>&1 | grep -i 'Overriding num_gpu_blocks'"
#   ssh -4 dgxspark1 "docker logs vllm_v026_256k 2>&1 | grep -i 'skipped memory profiling'"
#   ssh -4 dgxspark1 "docker logs vllm_v026_256k 2>&1 | grep -Ei 'KV cache size|Maximum concurrency'"
# Expect one "reserved ... for KV Cache as specified by kv_cache_memory_bytes config and skipped
# memory profiling" line per rank, and the head to report the mode's token count above.

set -uo pipefail

SSH_HOSTS=(dgxspark1 dgxspark2 dgxspark3 dgxspark4)
FABRIC_IPS=(10.42.0.11 10.42.0.12 10.42.0.13 10.42.0.14)
SSH_USER=fbsadmin
MASTER="${FABRIC_IPS[0]}"
IMAGE="fiosco/vllm:v0.2.0-arm64-nccl2.29.7"
NAME="vllm_v026_256k"
PORT=8210
MASTER_PORT=29501
HOST_HF=/data/huggingface
MODEL=/cache/huggingface/hub/glm52-quanttrio-unpruned
STOP_GRACE=150

# Measured from the 256k run's KV sizing log. block_size 64 x dcp_size 2 = 128 tok per block.
BYTES_PER_BLOCK=3504012
TOKENS_PER_BLOCK=128
GPU_MEM_UTIL=0.90

MODE=256k
DRYRUN=0
STOP=0
ACCEPT_UNVALIDATED=0

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31m! %s\033[0m\n' "$*" >&2; exit 1; }
on() { ssh -4 -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new "${SSH_USER}@$1" "${@:2}"; }

NNODES="${#SSH_HOSTS[@]}"

while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --dry-run) DRYRUN=1; shift;;
    --stop) STOP=1; shift;;
    --i-accept-unvalidated) ACCEPT_UNVALIDATED=1; shift;;
    *) die "unknown arg: $1 (use --mode 256k|288k|320k, --dry-run, --stop, --i-accept-unvalidated)";;
  esac
done

if [ "$STOP" = 1 ]; then
  say "graceful stop '${NAME}' on ${NNODES} nodes (SIGTERM, ${STOP_GRACE}s grace — never SIGKILL)"
  for h in "${SSH_HOSTS[@]}"; do
    on "$h" "docker stop -t ${STOP_GRACE} ${NAME} >/dev/null 2>&1; docker rm ${NAME} >/dev/null 2>&1; echo '   ${h}: stopped+removed'" \
      || echo "   $h: (nothing to stop / unreachable)"
  done
  exit 0
fi

case "$MODE" in
  256k) CTX=262144; KV_BYTES=7533625800; EVIDENCE="PROVEN — 36/36 HTTP 200 over 29m12s, 23.08 tok/s, 50.05% acceptance";;
  288k) CTX=294912; KV_BYTES=8479709040; EVIDENCE="UNVALIDATED — arithmetic only; +0.88 GiB/node, leaves dgxspark1 ~985 MiB";;
  320k) CTX=327680; KV_BYTES=9418784256; EVIDENCE="UNVALIDATED — arithmetic only; +1.76 GiB/node, leaves dgxspark1 ~89 MiB";;
  *) die "--mode must be 256k, 288k or 320k (got '${MODE}')";;
esac

BLOCKS=$(( KV_BYTES / BYTES_PER_BLOCK ))
KV_TOKENS=$(( BLOCKS * TOKENS_PER_BLOCK ))
[ $(( KV_BYTES % BYTES_PER_BLOCK )) -eq 0 ] \
  || die "internal: ${KV_BYTES} is not an exact multiple of ${BYTES_PER_BLOCK} B/block"
[ "$KV_TOKENS" -ge "$CTX" ] \
  || die "internal: ${KV_TOKENS} KV tokens < ${CTX} context — the pool cannot hold one full sequence"

if [ "$MODE" != 256k ] && [ "$ACCEPT_UNVALIDATED" = 0 ] && [ "$DRYRUN" = 0 ]; then
  die "mode ${MODE} has never been run on v0.26.0. Its numbers are arithmetic, not measured, and
   320k in particular leaves the head node ~89 MiB of headroom — the memory pressure the move to
   256k was made to escape. Re-run with --i-accept-unvalidated if that is genuinely intended,
   and not against a cluster that is serving."
fi

build_run() {
  local rank="$1" headless="$2" hostip="$3"
  local comp='{"cudagraph_mode":"FULL","max_cudagraph_capture_size":8}'
  local spec="{\"model\":\"${MODEL}\",\"method\":\"mtp\",\"quantization\":\"compressed-tensors\",\"num_speculative_tokens\":4,\"draft_sample_method\":\"probabilistic\"}"
  # FlashInferMLASparseSM120Impl derives from MLAAttentionImpl rather than
  # MLACommonBaseImpl, so it inherits no forward_mha and raises NotImplementedError
  # on any MHA-dispatched prefill. mla_attention.py picks MHA when a batch has
  # prefill tokens and prefill_max_seq_len <= index_topk (2048), while queries
  # <= reorder_batch_threshold (128 here: 64 heads / TP4 = 16) are classified as
  # decode. Prompts of 129-2048 tokens therefore kill the engine. Forcing MQA
  # keeps every prefill on the sparse path. Do not remove without first landing
  # forward_mha on the SM120 impl.
  local attn='{"sparse_mla_force_mqa":true}'
  local cmd=(
    docker run -d --name "$NAME"
    --entrypoint=
    --cap-add IPC_LOCK --ulimit memlock=-1:-1
    --network host --ipc host --shm-size 10gb --gpus all
    --device /dev/infiniband:/dev/infiniband
    -v "${HOST_HF}:/cache/huggingface"
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
    -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
    -e "HF_HOME=/cache/huggingface"
    -e "TRITON_CACHE_DIR=/cache/huggingface/.tritoncache-v026"
    -e "HF_HUB_OFFLINE=1"
    -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
    -e "TORCH_CUDA_ARCH_LIST=12.1a"
    -e "NCCL_NET=IB"
    -e "NCCL_IB_DISABLE=0"
    -e "NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1"
    -e "NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1"
    -e "GLOO_SOCKET_IFNAME=enp1s0f1np1"
    -e "NCCL_IB_GID_INDEX=3"
    -e "NCCL_CROSS_NIC=1"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_IGNORE_CPU_AFFINITY=1"
    -e "NCCL_DEBUG=WARN"
    -e "VLLM_HOST_IP=${hostip}"
    -e "NODE_RANK=${rank}"
    -e "MASTER_ADDR=${MASTER}"
    "$IMAGE"
    vllm serve "$MODEL"
    --served-model-name glm-5.2-quanttrio --host 0.0.0.0 --port "$PORT" --trust-remote-code
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --enable-prefix-caching
    --tensor-parallel-size 4 --pipeline-parallel-size 1
    --attention-backend FLASHINFER_MLA_SPARSE_SM120
    --attention-config "$attn"
    --max-model-len "$CTX" --max-num-seqs 1 --max-num-batched-tokens 4096
    --kv-cache-memory-bytes "$KV_BYTES"
    --gpu-memory-utilization "$GPU_MEM_UTIL" --kv-cache-dtype fp8_ds_mla
    --compilation-config "$comp" --async-scheduling --distributed-executor-backend mp
    --decode-context-parallel-size 2 --dcp-kv-cache-interleave-size 1
    --speculative-config "$spec"
    --nnodes "$NNODES" --node-rank "$rank" --master-addr "$MASTER" --master-port "$MASTER_PORT"
  )
  [ "$headless" = 1 ] && cmd+=(--headless)
  local out="" t
  for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done
  printf '%s' "${out# }"
}

say "GLM-5.2 QuantTrio on ${NNODES}x DGX Spark — mode ${MODE}"
echo "   image   ${IMAGE}   (NCCL 2.29.7 baked in: no LD_PRELOAD, no bind-mount)"
echo "   shape   TP4 / PP1 / DCP2 / MTP4 speculative, FLASHINFER_MLA_SPARSE_SM120, fp8_ds_mla"
echo "           async_scheduling, cudagraph_mode FULL, max-num-seqs 1"
printf '   KV      %s B per GPU = %.4f GiB -> %s blocks = %s tokens (%.4fx of %s ctx)\n' \
  "$KV_BYTES" "$(awk "BEGIN{print ${KV_BYTES}/1073741824}")" \
  "$BLOCKS" "$KV_TOKENS" "$(awk "BEGIN{print ${KV_TOKENS}/${CTX}}")" "$CTX"
echo "   sizing  --kv-cache-memory-bytes (per-GPU, skips profiling). NO --num-gpu-blocks-override."
echo "   util    ${GPU_MEM_UTIL} — startup gate only, ignored for KV sizing"
echo "   status  ${EVIDENCE}"
[ "$MODE" != 256k ] && printf '   \033[33mWARNING: %s is not the proven configuration. 256k is.\033[0m\n' "$MODE"
[ "$DRYRUN" = 1 ] && echo "   (dry-run — nothing will be executed)"

say "step 1: stop/remove any prior ${NAME} on all nodes (graceful, ${STOP_GRACE}s)"
for h in "${SSH_HOSTS[@]}"; do
  if [ "$DRYRUN" = 1 ]; then
    echo "  [dry] ${h}: docker stop -t ${STOP_GRACE} ${NAME}; docker rm ${NAME}"
  else
    printf '   %s: ' "$h"
    on "$h" "docker stop -t ${STOP_GRACE} ${NAME} >/dev/null 2>&1; docker rm ${NAME} >/dev/null 2>&1; echo ok" \
      || echo "(nothing to remove)"
  fi
done

say "step 2: launch (headless workers first, then the head)"
for ((rank=1; rank<NNODES; rank++)); do
  h="${SSH_HOSTS[$rank]}"
  run="$(build_run "$rank" 1 "${FABRIC_IPS[$rank]}")"
  if [ "$DRYRUN" = 1 ]; then
    printf '\n# worker %s rank=%d\nssh %s@%s %q\n' "$h" "$rank" "$SSH_USER" "$h" "$run"
  else
    printf '   worker %s rank=%d: ' "$h" "$rank"
    on "$h" "$run" || die "launch failed on ${h}"
  fi
done

run="$(build_run 0 0 "${FABRIC_IPS[0]}")"
if [ "$DRYRUN" = 1 ]; then
  printf '\n# head %s rank=0\nssh %s@%s %q\n' "${SSH_HOSTS[0]}" "$SSH_USER" "${SSH_HOSTS[0]}" "$run"
  exit 0
fi
printf '   head %s rank=0: ' "${SSH_HOSTS[0]}"
on "${SSH_HOSTS[0]}" "$run" || die "launch failed on the head"

say "launched mode ${MODE} — expect readiness ~9m40s after container create"
cat <<EOF
   no block override (expect ZERO matches):
     ssh -4 ${SSH_HOSTS[0]} "docker logs ${NAME} 2>&1 | grep -i 'Overriding num_gpu_blocks'"
   per-rank reservation (expect one line per rank):
     for h in ${SSH_HOSTS[*]}; do ssh -4 \$h "docker logs ${NAME} 2>&1 | grep -i 'skipped memory profiling'"; done
   pool (expect "GPU KV cache size: ${KV_TOKENS} tokens"):
     ssh -4 ${SSH_HOSTS[0]} "docker logs ${NAME} 2>&1 | grep -Ei 'KV cache size|Maximum concurrency'"
   memory:
     for h in ${SSH_HOSTS[*]}; do ssh -4 \$h 'free -m | head -3'; done
   serve:
     curl -s http://${SSH_HOSTS[0]}:${PORT}/v1/models
   logs:
     ssh -4 ${SSH_HOSTS[0]} docker logs -f ${NAME}
   stop:
     ./launch-fiosco-glm52-dgxspark.sh --stop
EOF
