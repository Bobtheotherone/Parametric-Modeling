#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"
log_dir="${VERIFY_LOG_DIR:-runs}"
log_path="${VERIFY_LOG_PATH:-${log_dir}/verify_strict.log}"

mkdir -p "${log_dir}"
"${python_bin}" -m tools.verify --strict 2>&1 | tee "${log_path}"
