#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALL_HPP="${ROOT_DIR}/include/tf_wrap/all.hpp"

if [[ ! -f "${ALL_HPP}" ]]; then
  echo "ERROR: ${ALL_HPP} not found"
  exit 1
fi

# Disallow any direct include of ops umbrella headers from all.hpp.
if grep -nE '^[[:space:]]*#include[[:space:]]*[<"]tf_wrap/ops(\.hpp|/all\.hpp|/)[>"]' "${ALL_HPP}"; then
  echo
  echo "ERROR: include/tf_wrap/all.hpp must NOT include ops headers."
  echo "       Ops must remain opt-in via #include <tf_wrap/ops.hpp> or <tf_wrap/ops/...>."
  exit 1
fi

# Also disallow including facade_ops from all.hpp.
if grep -nE '^[[:space:]]*#include[[:space:]]*[<"]tf_wrap/facade_ops\.hpp[>"]' "${ALL_HPP}"; then
  echo
  echo "ERROR: include/tf_wrap/all.hpp must NOT include tf_wrap/facade_ops.hpp."
  echo "       facade_ops.hpp is opt-in."
  exit 1
fi

echo "OK: all.hpp does not include ops headers."
