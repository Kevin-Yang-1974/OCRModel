#!/usr/bin/env bash
set -euo pipefail

training_run_id="${1:-layout_ablation_formal_v1_projector_only_seed42}"
dataset_id="${2:-formal_pdf_short_seed20260812}"
checkpoint_step="${3:-2000}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

checkpoint="${GOT_TRAINING_RUNS}/${training_run_id}/p2/model/checkpoint-${checkpoint_step}"
manifest="${GOT_LAYOUT_DATA}/${dataset_id}/validation/manifest.jsonl"
image_root="${GOT_LAYOUT_DATA}/${dataset_id}/validation"
for required in "${checkpoint}/model.safetensors" "${checkpoint}/config.json" "${manifest}"; do
    if [[ ! -f "${required}" ]]; then
        printf 'ERROR: missing required smoke input: %s\n' "${required}" >&2
        exit 66
    fi
done

stamp="$(date +%Y%m%d_%H%M%S)"
smoke_root="${GOT_EVALUATION_RUNS}/layout_ablation_selection_resume_smoke_${stamp}"
fixture="${smoke_root}/model"
output="${smoke_root}/selection"
mkdir -p "${fixture}"
for item in "${checkpoint}"/*; do
    name="$(basename -- "${item}")"
    if [[ "${name}" != "layout_training_metrics.json" ]]; then
        ln -s "${item}" "${fixture}/${name}"
    fi
done
python3 - "${fixture}/layout_training_metrics.json" "${checkpoint_step}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps({"global_step": int(sys.argv[2]), "ablation_id": "projector_only"}) + "\n",
    encoding="utf-8",
)
PY

selector=(
    "${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py"
    --ablation projector_only
    --model-root "${fixture}"
    --model-kind baseline
    --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}"
    --validation-manifest "${manifest}"
    --validation-image-root "${image_root}"
    --project-root "${ocrmodel_root}/src/GOT-OCR-2.0"
    --output-dir "${output}"
    --max-records 1
)
bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${selector[@]}" >/dev/null

candidate_log="${output}/step-$(printf '%08d' "${checkpoint_step}")/evaluator.log"
before_hash="$(sha256sum "${candidate_log}" | awk '{print $1}')"
case "${output}" in
    "${GOT_EVALUATION_RUNS}"/layout_ablation_selection_resume_smoke_*/selection) ;;
    *) printf 'ERROR: unsafe smoke output path: %s\n' "${output}" >&2; exit 64 ;;
esac
rm -f "${output}/selection.json" "${output}/SELECTION_FINISHED"
bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${selector[@]}" --resume >/dev/null
after_hash="$(sha256sum "${candidate_log}" | awk '{print $1}')"
if [[ "${before_hash}" != "${after_hash}" ]]; then
    printf 'ERROR: resume reran or modified the candidate evaluator log.\n' >&2
    exit 1
fi

python3 - "${smoke_root}" "${output}/selection.json" "${checkpoint_step}" <<'PY'
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
print(json.dumps({
    "event": "layout_ablation_selection_resume_smoke_completed",
    "run_root": sys.argv[1],
    "candidate_step": int(sys.argv[3]),
    "selected_step": selection["selected"]["optimizer_step"],
    "pages": selection["selected"]["validation_metrics"]["pages"],
    "evaluator_log_unchanged": True,
}, separators=(",", ":")))
PY
