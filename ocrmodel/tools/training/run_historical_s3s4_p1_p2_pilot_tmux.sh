#!/usr/bin/env bash
# Isolated historical-data P1 -> validation selection -> P2 pilot.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"
workspace_root="${OCR_WORKSPACE:-$(cd -- "${ocrmodel_root}/.." && pwd -P)}"
export OCR_WORKSPACE="${workspace_root}"
export GOT_LAYOUT_DATA="${GOT_LAYOUT_DATA:-${workspace_root}/training_data/got_layout_pages}"
export GOT_TRAINING_RUNS="${GOT_TRAINING_RUNS:-${workspace_root}/training_runs/GOT}"
export GOT_SOURCE_MODEL="${GOT_SOURCE_MODEL:-${workspace_root}/models/GOT-OCR2_0}"
export GOT_TOKENIZER_MODEL="${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}"
remote_root="${PILOT_REMOTE_ROOT:-/data3/yky/yangky_ocr_models}"
session="lavp_historical_s3s4_pilot_20260827_v1"
run_prefix="lavp_historical_s3s4_pilot_20260827_v1"
historical_root="${PILOT_HISTORICAL_ROOT:-${remote_root}/training_data/got_layout_pages/ancient_photo_diverse_formal_s3s4_20260826_v1}"
mthv2_root="${PILOT_MTHV2_ROOT:-${remote_root}/datasets/MTHv2/converted/mthv2_layout_page_v1}"
pilot_root="${PILOT_ROOT:-${remote_root}/training_runs/GOT/${run_prefix}}"
pilot_root_explicit=0
gpu_utilization_limit=50
distributed_strategy="deepspeed_zero2"
nccl_p2p_disable=1
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --run-prefix) run_prefix="$2"; shift 2 ;;
        --historical-root) historical_root="$2"; shift 2 ;;
        --mthv2-root) mthv2_root="$2"; shift 2 ;;
        --pilot-root) pilot_root="$2"; pilot_root_explicit=1; shift 2 ;;
        --gpu-utilization-limit) gpu_utilization_limit="$2"; shift 2 ;;
        --distributed-strategy) distributed_strategy="$2"; shift 2 ;;
        --nccl-p2p-disable) nccl_p2p_disable=1; shift ;;
        --session-inner) session_inner=1; shift ;;
        *) exit 64 ;;
    esac
done
if (( pilot_root_explicit == 0 )); then
    pilot_root="${remote_root}/training_runs/GOT/${run_prefix}"
fi
[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ && "${run_prefix}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ && "${gpu_utilization_limit}" -le 100 ]] || exit 64
[[ "${distributed_strategy}" == "deepspeed_zero2" || "${distributed_strategy}" == "ddp" ]] || exit 64
[[ "${historical_root}" == "${remote_root}/"* ]] || exit 66
[[ "${pilot_root}" == "${remote_root}/"* && "${pilot_root}" != "${remote_root}/" ]] || exit 66

got_runner=(bash "${ocrmodel_root}/tools/environment/run_got2.sh")
runner="${ocrmodel_root}/tools/training/run_variable_layout_a100.py"
selector="${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py"
project_root="${ocrmodel_root}/src/GOT-OCR-2.0"
train_manifest="${historical_root}/train/manifest.jsonl"
validation_manifest="${historical_root}/validation/manifest.jsonl"
test_manifest="${historical_root}/test/manifest.jsonl"
validation_root="${historical_root}/validation"
validation_lock="${pilot_root}/validation/pilot_validation_256.jsonl"
validation_meta="${pilot_root}/validation/pilot_validation_256.lock.json"

eligible_gpus() {
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits \
        | awk -F, -v limit="${gpu_utilization_limit}" '{gsub(/[[:space:]]/,"",$1); gsub(/[[:space:]]/,"",$2); if ($2 ~ /^[0-9]+$/ && $2 < limit) printf "%s,",$1}' \
        | sed 's/,$//' 
}

require_paths() {
    local path
    for path in "$train_manifest" "$validation_manifest" "$test_manifest" \
        "${historical_root}/train" "$validation_root" "${mthv2_root}/train/manifest.jsonl" \
        "${GOT_SOURCE_MODEL}/model.safetensors"; do
        [[ -e "$path" ]] || { printf '{"event":"historical_pilot_failed","error":"missing_path","path":"%s"}\n' "$path" >&2; return 1; }
    done
    [[ ! -e "${pilot_root}" ]] || { printf '{"event":"historical_pilot_failed","error":"pilot_output_exists","path":"%s"}\n' "$pilot_root" >&2; return 1; }
}

lock_validation() {
    mkdir -p "$(dirname -- "$validation_lock")"
    "${got_runner[@]}" - "$validation_manifest" "$validation_lock" "$validation_meta" <<'PY'
import hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
src,out,meta=map(Path,sys.argv[1:])
records=[json.loads(line) for line in src.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
if len(records)<256: raise SystemExit(f"validation manifest has only {len(records)} records")
def bucket(r):
    n=len(r.get("regions",[])); return "1-8" if n<=8 else "9-16" if n<=16 else "17-32" if n<=32 else "33-64" if n<=64 else "65-128" if n<=128 else ">128"
def direction(r):
    return "+".join(sorted({str(x.get("writing_direction","unknown")) for x in r.get("regions",[])})) or "unknown"
groups=defaultdict(list)
for r in records:
    if r.get("split")!="validation" or r.get("input_level")!="page": raise SystemExit("invalid validation record")
    groups[(str(r.get("tier","unknown")),bucket(r),direction(r))].append(r)
for values in groups.values(): values.sort(key=lambda r:str(r.get("page_id","")))
selected=[]; cursors={k:0 for k in sorted(groups)}
while len(selected)<256:
    progress=False
    for key in sorted(groups):
        i=cursors[key]
        if i<len(groups[key]):
            selected.append(groups[key][i]); cursors[key]=i+1; progress=True
            if len(selected)==256: break
    if not progress: raise SystemExit("unable to select 256 pages")
if len({r.get("page_id") for r in selected})!=256: raise SystemExit("duplicate selected page")
out.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in selected),encoding="utf-8")
payload={"status":"locked","selection_split":"validation","page_count":256,"source_manifest":str(src.resolve()),"manifest_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"selection_rule":"sorted tier x region_count_bucket x direction round-robin, page_id within stratum","tier_counts":dict(sorted(Counter(str(r.get("tier","unknown")) for r in selected).items())),"region_count_bucket_counts":dict(sorted(Counter(bucket(r) for r in selected).items())),"direction_counts":dict(sorted(Counter(direction(r) for r in selected).items())),"test_used_for_selection":False}
meta.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False,separators=(",",":")))
PY
}

run_pipeline() {
    require_paths
    mkdir -p "${pilot_root}"
    lock_validation
    "${got_runner[@]}" "${runner}" \
        --dataset-root "${historical_root}/train" --manifest "${train_manifest}" \
        --validation-manifest "${validation_lock}" --validation-image-root "${validation_root}" \
        --test-manifest "${test_manifest}" --source-model "${GOT_SOURCE_MODEL}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" --stages p1 \
        --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full --layout-memory-resolution 64 \
        --max-layout-records 512 --max-layout-tokens 2048 --p1-max-steps 2000 \
        --p2-max-steps 5000 --checkpoint-steps 1000 --checkpoint-retention 4 \
        --replay-manifest "${mthv2_root}/train/manifest.jsonl" --replay-image-root "${mthv2_root}/train" \
        --gpu-utilization-limit "${gpu_utilization_limit}" --distributed-strategy "${distributed_strategy}" \
        ${nccl_p2p_disable:+--nccl-p2p-disable} \
        --run-id "${run_prefix}_p1" --runs-root "${GOT_TRAINING_RUNS}" --project-root "${project_root}"
    local p1_selection="${GOT_TRAINING_RUNS}/${run_prefix}_p1/p1/validation_selection/selection.json"
    [[ -f "${p1_selection}" ]] || exit 1
    local selected_model
    selected_model="$("${got_runner[@]}" - "$p1_selection" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["selected"]["model_path"])
PY
)"
    "${got_runner[@]}" "${runner}" \
        --dataset-root "${historical_root}/train" --manifest "${train_manifest}" \
        --validation-manifest "${validation_lock}" --validation-image-root "${validation_root}" \
        --test-manifest "${test_manifest}" --source-model "${selected_model}" \
        --source-validation-selection "${p1_selection}" --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --stages p2 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
        --layout-memory-resolution 64 --max-layout-records 512 --max-layout-tokens 2048 \
        --p2-max-steps 5000 --checkpoint-steps 1000 --checkpoint-retention 6 \
        --gpu-utilization-limit "${gpu_utilization_limit}" --distributed-strategy "${distributed_strategy}" \
        ${nccl_p2p_disable:+--nccl-p2p-disable} \
        --run-id "${run_prefix}_p2" --runs-root "${GOT_TRAINING_RUNS}" --project-root "${project_root}"
    mkdir -p "${pilot_root}/p2/validation_selection"
    "${got_runner[@]}" "${selector}" --ablation vlqa_layout_p1_p2 --model-root "${GOT_TRAINING_RUNS}/${run_prefix}_p2/p2/model" \
        --model-kind pvld --selection-purpose ocr --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --validation-manifest "${validation_lock}" --validation-image-root "${validation_root}" \
        --output-dir "${pilot_root}/p2/validation_selection" --project-root "${project_root}" \
        --max-regions 512 --max-records 0 --max-new-tokens 2048 --parallel-gpu-ids "$(eligible_gpus)" \
        --gpu-utilization-limit "${gpu_utilization_limit}"
    printf '{"event":"historical_pilot_completed","run_prefix":"%s","validation_lock":"%s","p1_selection":"%s","p2_selection":"%s","test_run":false,"p3_run":false}\n' "$run_prefix" "$validation_meta" "$p1_selection" "${pilot_root}/p2/validation_selection/selection.json"
}

if (( session_inner == 1 )); then run_pipeline; exit; fi
command -v tmux >/dev/null 2>&1 || exit 69
[[ -x "${ocrmodel_root}/tools/environment/run_got2.sh" ]] || exit 66
tmux has-session -t "${session}" 2>/dev/null && exit 73
log_root="${GOT_TRAINING_RUNS}/${run_prefix}_pilot_logs"; mkdir -p "$log_root"
log_path="${log_root}/${session}.log"; script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --run-prefix '${run_prefix}' --historical-root '${historical_root}' --mthv2-root '${mthv2_root}' --pilot-root '${pilot_root}' --gpu-utilization-limit '${gpu_utilization_limit}' --distributed-strategy '${distributed_strategy}' --nccl-p2p-disable >'${log_path}' 2>&1"
sleep 3; tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "$log_path" >&2 || true; exit 1; }
printf '{"event":"historical_pilot_armed","session":"%s","run_prefix":"%s","distributed_strategy":"%s","nccl_p2p_disable":true,"p1_steps":2000,"p2_steps":5000,"checkpoint_steps":1000,"test_run":false,"p3_run":false,"log":"%s"}\n' "$session" "$run_prefix" "$distributed_strategy" "$log_path"
