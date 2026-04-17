#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lock_file="$repo_root/submodules.lock"

sync_git_checkout() {
  local paths=(
    "external/fyp-power-measure"
    "external/stm32ai-modelzoo"
    "external/stm32ai-modelzoo-services"
    "external/ultralytics"
    "external/TinyissimoYOLO"
  )

  local path
  for path in "${paths[@]}"; do
    git -C "$repo_root" submodule sync "$path"
  done

  for path in "${paths[@]}"; do
    git -C "$repo_root" -c submodule.recurse=false submodule update --init --checkout -- "$path"
  done
}

restore_from_lock() {
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to restore stripped submodules." >&2
    exit 1
  fi

  if [ ! -f "$lock_file" ]; then
    echo "Missing $lock_file. Use the publish script to create a submission archive first." >&2
    exit 1
  fi

  while IFS=$'\t' read -r path url commit mode; do
    [ -n "${path:-}" ] || continue
    [ "${path#\#}" = "$path" ] || continue
    _="$mode"

    local_target="$repo_root/$path"
    mkdir -p "$(dirname "$local_target")"

    if [ -d "$local_target/.git" ]; then
      :
    elif [ -e "$local_target" ] && [ -n "$(find "$local_target" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
      echo "Refusing to overwrite non-empty path: $path" >&2
      exit 1
    else
      rm -rf "$local_target"
      git clone --filter=blob:none --no-checkout "$url" "$local_target"
    fi

    git -C "$local_target" checkout --force "$commit"
  done < "$lock_file"
}

if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  sync_git_checkout
else
  restore_from_lock
fi
