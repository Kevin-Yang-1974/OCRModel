#!/usr/bin/env bash
# Run one q32 GLMOCR comparison arm on the A100 host.
# The historical launcher names mention MTHv2; this wrapper supplies only the
# portable Dunhuang/local-gazetteer compatibility view and never reads MTHv2.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
dataset_parent="${GLMOCR_Q32_DATASET_PARENT:-/data3/yky/yangky_ocr_models/datasets}"
archive="${GLMOCR_Q32_ARCHIVE:-${dataset_parent}/dunhuang_local_gazetteer_q32_v1_portable.zip}"
portable_root="${GLMOCR_Q32_PORTABLE_ROOT:-${dataset_parent}/dunhuang_local_gazetteer_q32_v1_portable}"
dataset_root="${GLMOCR_Q32_DATASET_ROOT:-${dataset_parent}/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"
code="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"

run_id="${1:-}"
mode="${2:-}"
[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '{"event":"glmocr_q32_a100_failed","error":"invalid_run_id"}\n' >&2
    exit 64
}
case "${mode}" in
    geometry|attention) auxiliary_weight="${GLMOCR_A100_AUXILIARY_WEIGHT:-0.4}" ;;
    content_only) auxiliary_weight="0.0" ;;
    *)
        printf '{"event":"glmocr_q32_a100_failed","error":"invalid_comparison_mode","mode":"%s"}\n' "${mode}" >&2
        exit 64
        ;;
esac

prepare_dataset_view() {
    mkdir -p "${dataset_parent}"
    if [[ ! -d "${portable_root}" ]]; then
        [[ -f "${archive}" ]] || {
            printf '{"event":"glmocr_q32_a100_failed","error":"portable_archive_missing","path":"%s"}\n' "${archive}" >&2
            exit 66
        }
        staging="${dataset_parent}/.dunhuang_local_gazetteer_q32_v1_extract"
        [[ ! -e "${staging}" ]] || {
            printf '{"event":"glmocr_q32_a100_failed","error":"stale_extract_staging","path":"%s"}\n' "${staging}" >&2
            exit 74
        }
        mkdir -p "${staging}"
        unzip -q "${archive}" -d "${staging}"
        extracted="${staging}/dunhuang_local_gazetteer_q32_v1_portable"
        [[ -d "${extracted}" ]] || {
            printf '{"event":"glmocr_q32_a100_failed","error":"unexpected_archive_layout"}\n' >&2
            exit 66
        }
        [[ ! -e "${portable_root}" ]] || {
            printf '{"event":"glmocr_q32_a100_failed","error":"portable_root_appeared_during_prepare"}\n' >&2
            exit 74
        }
        mv -- "${extracted}" "${portable_root}"
    fi

    for split in train validation test; do
        [[ -f "${portable_root}/manifests/${split}.jsonl" ]] || {
            printf '{"event":"glmocr_q32_a100_failed","error":"portable_manifest_missing","split":"%s"}\n' "${split}" >&2
            exit 66
        }
    done
    [[ -f "${portable_root}/protocol.json" && -d "${portable_root}/images" ]] || {
        printf '{"event":"glmocr_q32_a100_failed","error":"portable_protocol_or_images_missing"}\n' >&2
        exit 66
    }

    mkdir -p "${dataset_root}"
    for split in train validation test; do
        split_dir="${dataset_root}/${split}"
        mkdir -p "${split_dir}"
        manifest="${split_dir}/manifest.jsonl"
        if [[ -f "${manifest}" ]]; then
            cmp -s "${portable_root}/manifests/${split}.jsonl" "${manifest}" || {
                printf '{"event":"glmocr_q32_a100_failed","error":"compat_manifest_mismatch","split":"%s"}\n' "${split}" >&2
                exit 74
            }
        else
            cp -- "${portable_root}/manifests/${split}.jsonl" "${manifest}"
        fi
        image_link="${split_dir}/images"
        if [[ -e "${image_link}" && ! -L "${image_link}" ]]; then
            printf '{"event":"glmocr_q32_a100_failed","error":"compat_image_path_not_symlink","split":"%s"}\n' "${split}" >&2
            exit 74
        fi
        if [[ ! -L "${image_link}" ]]; then
            ln -s "${portable_root}/images" "${image_link}"
        fi
        [[ "$(readlink -f -- "${image_link}")" == "$(readlink -f -- "${portable_root}/images")" ]] || {
            printf '{"event":"glmocr_q32_a100_failed","error":"compat_image_symlink_mismatch","split":"%s"}\n' "${split}" >&2
            exit 74
        }
    done
    printf '{"event":"glmocr_q32_dataset_ready","dataset_root":"%s","portable_root":"%s","train_manifest":"%s","validation_manifest":"%s","test_manifest":"%s"}\n' \
        "${dataset_root}" "${portable_root}" \
        "${dataset_root}/train/manifest.jsonl" "${dataset_root}/validation/manifest.jsonl" "${dataset_root}/test/manifest.jsonl"
}

prepare_dataset_view

export GLMOCR_A100_ROOT="${remote_root}"
export GLMOCR_A100_CODE_ROOT="${code}"
export GLMOCR_A100_MTHV2_ROOT="${dataset_root}"
export GLMOCR_A100_MODE="${mode}"
export GLMOCR_A100_AUXILIARY_WEIGHT="${auxiliary_weight}"
export GLMOCR_A100_DATASET_LABEL=dunhuang_local_gazetteer_q32_v1
export GLMOCR_A100_PROTOCOL_LABEL=glm_ocr_dunhuang_local_gazetteer_group_isolated_v1
export GLMOCR_A100_ALLOW_COUNT_MISMATCH=1
export GLMOCR_A100_NUM_QUERIES=32
export GLMOCR_A100_MAX_STEPS="${GLMOCR_A100_MAX_STEPS:-2000}"
export GLMOCR_A100_LR_SCHEDULE_STEPS="${GLMOCR_A100_LR_SCHEDULE_STEPS:-${GLMOCR_A100_MAX_STEPS}}"
export GLMOCR_A100_VALIDATION_INTERVAL="${GLMOCR_A100_VALIDATION_INTERVAL:-500}"
export GLMOCR_A100_MAX_EVAL_NEW_TOKENS="${GLMOCR_A100_MAX_EVAL_NEW_TOKENS:-1536}"
export GLMOCR_A100_GENERATION_MODE="${GLMOCR_A100_GENERATION_MODE:-loop_recovery}"
export GLMOCR_A100_GPU_IDS="${GLMOCR_A100_GPU_IDS:-0,1,2,3,4}"
export GLMOCR_A100_SEED="${GLMOCR_A100_SEED:-42}"

exec bash "${code}/tools/training/run_glmocr_a100_decoder_lora.sh" "${run_id}"
