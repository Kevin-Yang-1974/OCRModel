#!/usr/bin/env bash
set -euo pipefail

# Reinstall only the selected project environment. This script never clones code,
# downloads models, copies datasets, or changes system Python/CUDA.
umask 027

target="${1:-got}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
workspace="${OCR_WORKSPACE:-$(cd -- "${ocrmodel_root}/.." && pwd -P)}"
got_source="${GOT_PROJECT_ROOT:-${ocrmodel_root}/src/GOT-OCR-2.0}"
got_env="${workspace}/envs/got2"
ananda_env="${workspace}/envs/anandasky"
micromamba="${workspace}/.tools/micromamba"
temporary_dir=""
temporary_base="${TMPDIR:-/tmp}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [[ -n "${temporary_dir}" && -d "${temporary_dir}" ]]; then
        case "${temporary_dir}" in
            "${temporary_base%/}"/ocrmodel-env.*) rm -rf -- "${temporary_dir}" ;;
            *) printf 'ERROR: refusing to remove unexpected temporary path: %s\n' "${temporary_dir}" >&2 ;;
        esac
    fi
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

ensure_environment() {
    local env_dir="$1"
    local expected_python="$2"
    local actual_python

    if [[ -x "${env_dir}/bin/python" ]]; then
        actual_python="$("${env_dir}/bin/python" -c 'import platform; print(platform.python_version())')"
        [[ "${actual_python}" == "${expected_python}" ]] || \
            die "Existing environment ${env_dir} uses Python ${actual_python}, expected ${expected_python}."
        return
    fi

    [[ ! -e "${env_dir}" ]] || die "Incomplete environment path already exists: ${env_dir}"
    [[ -x "${micromamba}" ]] || die "Missing Micromamba: ${micromamba}. Reuse the existing server installation or provision it separately."
    "${micromamba}" create -y -p "${env_dir}" -c conda-forge "python=${expected_python}" pip
}

install_got() {
    local python="${got_env}/bin/python"
    local lock_file="${script_dir}/requirements-got-server.lock.txt"
    local filtered_lock="${temporary_dir}/got.filtered.txt"

    [[ -f "${got_source}/pyproject.toml" ]] || die "Missing synchronized GOT source: ${got_source}"
    [[ -f "${lock_file}" ]] || die "Missing GOT lock file: ${lock_file}"
    "${python}" -m pip install "pip==26.1.2" "setuptools==83.0.0" "wheel==0.47.0"
    "${python}" -m pip install --extra-index-url "https://download.pytorch.org/whl/cu118" \
        "torch==2.0.1+cu118" "torchvision==0.15.2+cu118"
    grep -Ev '^(GOT|pip|setuptools|torch|torchvision|wheel)==' "${lock_file}" > "${filtered_lock}"
    "${python}" -m pip install --no-deps -r "${filtered_lock}"
    "${python}" -m pip install --no-deps -e "${got_source}"
    "${python}" -m pip check
}

install_anandasky() {
    local python="${ananda_env}/bin/python"
    local lock_file="${script_dir}/requirements-anandasky-server.lock.txt"
    local filtered_lock="${temporary_dir}/anandasky.filtered.txt"

    [[ -f "${lock_file}" ]] || die "Missing AnandaSky lock file: ${lock_file}"
    "${python}" -m pip install "pip==26.1.2" "setuptools==83.0.0" "wheel==0.47.0"
    "${python}" -m pip install --extra-index-url "https://download.pytorch.org/whl/cu121" \
        "torch==2.5.1+cu121"
    # FlashAttention uses the separately verified binary-wheel installer below.
    grep -Ev '^(flash[-_]attn|pip|setuptools|torch|wheel)==' "${lock_file}" > "${filtered_lock}"
    "${python}" -m pip install --no-deps -r "${filtered_lock}"
    "${python}" -m pip check
}

main() {
    [[ -n "${workspace}" && "${workspace}" != "/" ]] || die "Unsafe OCR_WORKSPACE: ${workspace}"
    case "${target}" in
        got|anandasky|all) ;;
        *) die "Usage: $0 [got|anandasky|all]" ;;
    esac

    for command_name in grep mktemp; do
        require_command "${command_name}"
    done
    temporary_dir="$(mktemp -d "${temporary_base%/}/ocrmodel-env.XXXXXX")"
    trap cleanup EXIT

    mkdir -p "${workspace}/cache/pip" "${workspace}/cache/huggingface" "${workspace}/envs"
    export MAMBA_ROOT_PREFIX="${workspace}/.micromamba"
    export PIP_CACHE_DIR="${workspace}/cache/pip"
    export HF_HOME="${workspace}/cache/huggingface"
    export PYTHONNOUSERSITE=1
    export TOKENIZERS_PARALLELISM=false

    if [[ "${target}" == "got" || "${target}" == "all" ]]; then
        ensure_environment "${got_env}" "3.10.20"
        install_got
    fi
    if [[ "${target}" == "anandasky" || "${target}" == "all" ]]; then
        ensure_environment "${ananda_env}" "3.11.15"
        install_anandasky
    fi

    printf 'SETUP_OK target=%s workspace=%s\n' "${target}" "${workspace}"
}

main "$@"
