#!/usr/bin/env bash
set -u
export PYTHONIOENCODING=utf-8 UV_LINK_MODE=copy
LOG="results/raw/epoch2_bootstrap.log"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

log "=== Phase 1: re-evaluate epoch-1 artifacts under criteria v2 ==="
n=0; ok=0; failed=0
for dir in results/artifacts/L*_gemini_*/run-*/; do
  dir="${dir%/}"
  n=$((n + 1))
  uv run benchmark-agent evaluate --run "$dir" >>"$LOG.detail" 2>&1
  verdict=$(uv run python - "$dir" <<'EOF'
import json, sys, pathlib
s = json.loads((pathlib.Path(sys.argv[1]) / "evaluation" / "summary.json").read_text(encoding="utf-8"))
succ = s.get("success") or {}
conf = s.get("conformance") or {}
bad = [f"{c['operation']}/{c['name']}" for c in conf.get("checks", []) if not c.get("ok")]
print(f"status={s.get('status')} success={all(succ.get(k) for k in ('syntax_ok','executed_ok','semantics_ok'))} conformance={conf.get('ok')} failed_checks={bad} deviations={len(conf.get('deviations', []))}")
EOF
)
  case "$verdict" in *"success=True"*) ok=$((ok + 1));; *) failed=$((failed + 1));; esac
  log "re-eval $n $dir -> $verdict"
done
log "Phase 1 done: $n runs re-evaluated, $ok success, $failed not"

log "=== Phase 2: L1 x3 with pinned workload (epoch 2) ==="
uv run benchmark-agent run --condition L1_gemini_zeroshot --repetitions 3 2>&1 | grep -E "^(L1_|Runs:|Matrix run failed|Outcomes)" | tee -a "$LOG"

log "=== Phase 3: temperature 0.7 + per-repetition seed pilot, L2 x5 (epoch 2) ==="
uv run benchmark-agent run --condition L2_gemini_zeroshot --repetitions 5 2>&1 | grep -E "^(L2_|Runs:|Matrix run failed|Outcomes)" | tee -a "$LOG"

log "=== Dashboard ==="
uv run benchmark-agent dashboard 2>&1 | head -2 | tee -a "$LOG"
log "BOOTSTRAP DONE"
