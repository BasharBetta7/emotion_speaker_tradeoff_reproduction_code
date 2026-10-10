#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
paper_root="$repo_root/paper/latex"

track="all"
skip_build=0

usage() {
  cat <<USAGE
Usage: $0 [--track conference|access|all] [--skip-build]

Runs consistency checks for paper LaTeX tracks.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track)
      track="$2"
      shift 2
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$track" != "conference" && "$track" != "access" && "$track" != "all" ]]; then
  echo "Invalid track: $track" >&2
  exit 1
fi

tracks=("$track")
if [[ "$track" == "all" ]]; then
  tracks=(conference access)
fi

if [[ "$skip_build" -eq 0 ]]; then
  "$repo_root/scripts/build_paper_latex.sh" --track "$track"
fi

status=0

required_artifacts=(
  "paper/artifacts/tradeoff_analysis/alpha_sweep_metrics.png"
  "paper/artifacts/tradeoff_analysis/tradeoff_frontier.png"
  "paper/artifacts/tradeoff_analysis/per_emotion_tradeoff.png"
  "paper/artifacts/tradeoff_analysis/per_speaker_spread.png"
  "paper/artifacts/tradeoff_analysis/checkpoint_comparison.png"
  "paper/artifacts/tradeoff_analysis/main_operating_points.csv"
  "paper/artifacts/tradeoff_analysis/per_emotion_alpha_1_0.csv"
  "paper/artifacts/tradeoff_analysis/checkpoint_summary.csv"
  "paper/artifacts/tradeoff_analysis/ser_reliability_summary.csv"
)

for rel in "${required_artifacts[@]}"; do
  if [[ ! -f "$repo_root/$rel" ]]; then
    echo "[ERROR] Missing artifact: $rel" >&2
    status=1
  fi
done

extract_graphics_paths() {
  local t="$1"
  local track_dir="$paper_root/$t"
  rg -No "\\\\includegraphics(?:\\[[^\\]]*\\])?\\{([^}]*)\\}" "$track_dir/main.tex" "$paper_root/shared/sections"/*.tex \
    | sed -E 's/.*\{([^}]*)\}.*/\1/' \
    | sort -u
}

for t in "${tracks[@]}"; do
  track_dir="$paper_root/$t"
  while IFS= read -r relpath; do
    [[ -z "$relpath" ]] && continue
    if [[ "$relpath" == http* ]]; then
      continue
    fi
    if [[ ! -f "$track_dir/$relpath" ]]; then
      echo "[ERROR] Missing figure path for $t: $relpath" >&2
      status=1
    fi
  done < <(extract_graphics_paths "$t")

done

for t in "${tracks[@]}"; do
  log_file="$paper_root/build/$t/main.log"
  if [[ ! -f "$log_file" ]]; then
    echo "[ERROR] Missing log file: $log_file" >&2
    status=1
    continue
  fi

  if rg -n "! LaTeX Error:|Undefined control sequence|Emergency stop|Fatal error occurred|Citation .* undefined|Reference .* undefined|There were undefined references" "$log_file" >/dev/null; then
    echo "[ERROR] Blocking LaTeX issue detected in $log_file" >&2
    rg -n "! LaTeX Error:|Undefined control sequence|Emergency stop|Fatal error occurred|Citation .* undefined|Reference .* undefined|There were undefined references" "$log_file" >&2 || true
    status=1
  fi

  if rg -n "Overfull \\\\hbox|Underfull \\\\hbox|LaTeX Warning" "$log_file" >/dev/null; then
    echo "[WARN] Non-blocking warnings in $log_file" >&2
    rg -n "Overfull \\\\hbox|Underfull \\\\hbox|LaTeX Warning" "$log_file" >&2 || true
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "Paper LaTeX checks failed." >&2
  exit "$status"
fi

echo "Paper LaTeX checks passed for track: $track"
