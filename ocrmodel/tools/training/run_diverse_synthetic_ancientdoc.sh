#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_diverse_synthetic_ancientdoc.sh train-synthetic --dataset-id ID --run-prefix NAME [GPU options]
  run_diverse_synthetic_ancientdoc.sh train-ancient-core --synthetic-selection FILE --run-prefix NAME [options]
  run_diverse_synthetic_ancientdoc.sh select-c4 --c4-run DIR --gpu-ids IDS [options]
  run_diverse_synthetic_ancientdoc.sh train-replay --c4-selection FILE --run-prefix NAME [options]
  run_diverse_synthetic_ancientdoc.sh select-ancient --training-suite DIR --c4-selection FILE [options]
  run_diverse_synthetic_ancientdoc.sh test-ancient --selection FILE --c4-selection FILE [options]
  run_diverse_synthetic_ancientdoc.sh smoke [--dataset-id ID] [--gpu-id ID]

Protocol defaults:
  synthetic A5 P1/P2: 12,000/24,000 optimizer steps, checkpoint every 2,000;
  AncientDoc core/replay: 12,000 steps, checkpoint every 2,000;
  C5/C6 primary:synthetic replay: 7:1 (12.5% requested replay).

train-synthetic accepts --p1-steps, --p2-steps and --checkpoint-steps overrides.
It also accepts --gpu-utilization-limit (default 50); a target GPU is allowed
only when its instantaneous utilization is strictly below that limit.

Every selection stage uses validation only. test-ancient is separate and requires the
frozen selection.json, so it cannot silently select on test. Existing outputs are not overwritten.
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
command="${1:-}"
if [[ -z "${command}" || "${command}" == "--help" || "${command}" == "-h" ]]; then
    usage
    exit 0
fi
shift

case "${command}" in
    train-synthetic)
        p1_steps=12000
        p2_steps=24000
        checkpoint_steps=2000
        gpu_utilization_limit=50
        forwarded=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --p1-steps) p1_steps="${2:-}"; shift 2 ;;
                --p2-steps) p2_steps="${2:-}"; shift 2 ;;
                --checkpoint-steps) checkpoint_steps="${2:-}"; shift 2 ;;
                --gpu-utilization-limit) gpu_utilization_limit="${2:-}"; shift 2 ;;
                *) forwarded+=("$1"); shift ;;
            esac
        done
        exec bash "${script_dir}/run_layout_ablation_suite.sh" \
            --ablations vlqa_layout_p1_p2 \
            --p1-steps "${p1_steps}" \
            --p2-steps "${p2_steps}" \
            --checkpoint-steps "${checkpoint_steps}" \
            --layout-loss-preset layout_full \
            --selection-only \
            --gpu-utilization-limit "${gpu_utilization_limit}" \
            "${forwarded[@]}"
        ;;
    train-ancient-core)
        synthetic_selection=""
        forwarded=()
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --synthetic-selection) synthetic_selection="${2:-}"; shift 2 ;;
                *) forwarded+=("$1"); shift ;;
            esac
        done
        if [[ -z "${synthetic_selection}" ]]; then
            printf 'ERROR: train-ancient-core requires --synthetic-selection.\n' >&2
            exit 64
        fi
        selected_model="$(python3 - "${synthetic_selection}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]).expanduser().resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "ok" or payload.get("ablation_id") != "vlqa_layout_p1_p2":
    raise SystemExit("synthetic selection must be status=ok for vlqa_layout_p1_p2")
if payload.get("selection_split") != "validation" or payload.get("test_used_for_selection") is not False:
    raise SystemExit("synthetic checkpoint must be selected on validation only")
model = Path(payload["selected"]["model_path"]).resolve()
if not model.is_dir():
    raise SystemExit(f"selected synthetic model is missing: {model}")
print(model)
PY
)"
        exec bash "${script_dir}/run_ancientdoc.sh" train-core \
            --steps 12000 --checkpoint-steps 2000 \
            --p2-model "${selected_model}" --p2-selection "${synthetic_selection}" \
            "${forwarded[@]}"
        ;;
    select-c4)
        exec bash "${script_dir}/run_ancientdoc.sh" select-c4 "$@"
        ;;
    train-replay)
        exec bash "${script_dir}/run_ancientdoc.sh" train-replay \
            --steps 12000 --checkpoint-steps 2000 --primary-per-replay 7 "$@"
        ;;
    select-ancient)
        exec bash "${ocrmodel_root}/tools/evaluation/run_ancientdoc.sh" \
            --phase select "$@"
        ;;
    test-ancient)
        exec bash "${ocrmodel_root}/tools/evaluation/run_ancientdoc.sh" \
            --phase test "$@"
        ;;
    smoke)
        dataset_id="formal_pdf_short_seed20260812"
        gpu_id="${GOT_PHYSICAL_GPU:-0}"
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --dataset-id) dataset_id="${2:-}"; shift 2 ;;
                --gpu-id) gpu_id="${2:-}"; shift 2 ;;
                *) printf 'ERROR: unknown smoke argument: %s\n' "$1" >&2; exit 64 ;;
            esac
        done
        bash "${script_dir}/run_layout_ablation_smoke.sh" \
            --dataset-id "${dataset_id}" \
            --ablations vlqa_layout_p1_p2 \
            --gpu-id "${gpu_id}"
        bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
            - "${ocrmodel_root}/src/GOT-OCR-2.0" <<'PY'
import json, sys
from pathlib import Path
project = Path(sys.argv[1])
sys.path.insert(0, str(project))
sys.path.insert(0, str(project / "scripts"))
from layout_page_dataset import InterleavedLayoutDataset
class D:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return i
dataset = InterleavedLayoutDataset(D(70), D(11), primary_per_replay=7)
values = [dataset[i] for i in range(16)]
assert len(dataset) == 80
assert values[7] == 0 and values[15] == 1
print(json.dumps({"event":"diverse_synthetic_ancientdoc_smoke_completed","primary_per_replay":7,"requested_replay_fraction":0.125}, separators=(",", ":")))
PY
        ;;
    *)
        printf 'ERROR: unknown command: %s\n' "${command}" >&2
        usage
        exit 64
        ;;
esac
