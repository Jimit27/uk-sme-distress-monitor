#!/usr/bin/env bash
# Run a pipeline command, keep its log, and on failure surface the last lines
# as a GitHub Actions error annotation (visible in the checks API and UI).
set -o pipefail
name="$1"; shift
mkdir -p logs
"$@" 2>&1 | tee "logs/${name}.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  msg=$(tail -n 25 "logs/${name}.log" | sed 's/%/%25/g' | sed ':a;N;$!ba;s/\n/%0A/g')
  echo "::error title=${name} failed::${msg}"
fi
exit "$status"
