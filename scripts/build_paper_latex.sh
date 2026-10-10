#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
paper_root="$repo_root/paper/latex"

track="all"
clean=0

usage() {
  cat <<USAGE
Usage: $0 [--track conference|access|draft|all] [--clean]

Build LaTeX outputs for paper tracks.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track)
      track="$2"
      shift 2
      ;;
    --clean)
      clean=1
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

if [[ "$track" != "conference" && "$track" != "access" && "$track" != "draft" && "$track" != "all" ]]; then
  echo "Invalid track: $track" >&2
  exit 1
fi

clean_track() {
  local t="$1"
  local track_dir="$paper_root/$t"
  rm -rf "$paper_root/build/$t"
  if command -v latexmk >/dev/null 2>&1; then
    (cd "$track_dir" && latexmk -C main.tex >/dev/null 2>&1 || true)
  fi
  rm -f "$track_dir"/main.{aux,bbl,blg,fdb_latexmk,fls,log,out,pdf,toc}
}

build_one() {
  local t="$1"
  local track_dir="$paper_root/$t"
  local out_dir="$paper_root/build/$t"
  local main_tex="$track_dir/main.tex"
  local engine="pdflatex"

  mkdir -p "$out_dir"

  if [[ -f "$main_tex" ]]; then
    local tex_program
    tex_program="$(grep -m1 -E '^% !TEX program = ' "$main_tex" | sed 's/^% !TEX program = //')"
    case "$tex_program" in
      lualatex|xelatex|pdflatex)
        engine="$tex_program"
        ;;
    esac
  fi

  if command -v latexmk >/dev/null 2>&1; then
    (
      cd "$track_dir"
      case "$engine" in
        lualatex)
          latexmk -lualatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
          ;;
        xelatex)
          latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
          ;;
        *)
          latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
          ;;
      esac
    )
  else
    (
      cd "$track_dir"
      "$engine" -interaction=nonstopmode -halt-on-error -file-line-error main.tex
      if [[ -f main.aux ]] && grep -Eq '\\citation|\\bibdata' main.aux; then
        bibtex main
      fi
      "$engine" -interaction=nonstopmode -halt-on-error -file-line-error main.tex
      "$engine" -interaction=nonstopmode -halt-on-error -file-line-error main.tex
    )
  fi

  cp -f "$track_dir"/main.{pdf,log,aux} "$out_dir"/
  [[ -f "$track_dir/main.bbl" ]] && cp -f "$track_dir/main.bbl" "$out_dir"/ || true
  [[ -f "$track_dir/main.blg" ]] && cp -f "$track_dir/main.blg" "$out_dir"/ || true

  echo "Built $t PDF: $out_dir/main.pdf"
}

tracks=("$track")
if [[ "$track" == "all" ]]; then
  tracks=(conference access)
fi

if [[ "$clean" -eq 1 ]]; then
  for t in "${tracks[@]}"; do
    clean_track "$t"
  done
  exit 0
fi

for t in "${tracks[@]}"; do
  build_one "$t"
done
