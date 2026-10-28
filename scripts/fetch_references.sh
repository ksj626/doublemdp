#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
fetch() {
  local name="$1" url="$2" commit="$3"
  if [[ ! -d "third_party/$name/.git" ]]; then
    git clone "$url" "third_party/$name"
  fi
  if [[ -n "$(git -C "third_party/$name" status --porcelain)" ]]; then
    echo "Reference $name has local changes; preserve and review them before checkout." >&2
    return 1
  fi
  git -C "third_party/$name" fetch origin "$commit"
  git -C "third_party/$name" checkout --detach "$commit"
  test "$(git -C "third_party/$name" rev-parse HEAD)" = "$commit"
}
fetch rql https://github.com/aoberai/rql.git 229c956efb4494c2b9bb0bbddbd67b761c93f1cc
fetch flow-rl https://github.com/typoverflow/flow-rl.git bae22f4cad3f9ce0a220aceb877b0e979de89808
fetch dppo https://github.com/irom-princeton/dppo.git dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc
fetch ogbench https://github.com/seohongpark/ogbench.git 572788d97c2a6225b8edb7c420e304c04d1b3459
