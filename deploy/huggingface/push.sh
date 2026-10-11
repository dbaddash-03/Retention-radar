#!/usr/bin/env bash
# Usage: deploy/huggingface/push.sh <hf-username>/<space-name>
# Prereq: create an empty Space (SDK: Docker, blank template) on huggingface.co, and be logged in
# (git will ask for your HF username and an access token with "write" scope as the password).
set -euo pipefail
SPACE="${1:?usage: push.sh <hf-username>/<space-name>}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d)"
git clone "https://huggingface.co/spaces/${SPACE}" "$TMP/space"
cd "$TMP/space"
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cd "$HERE"
git ls-files -z | grep -zv '^deploy/' | xargs -0 -I{} cp --parents {} "$TMP/space/"
cp deploy/huggingface/README.md "$TMP/space/README.md"
cd "$TMP/space"
git add -A
git commit -m "Deploy Retention Radar" || echo "nothing to commit"
git push
echo "Done. Build logs: https://huggingface.co/spaces/${SPACE}?logs=build"
