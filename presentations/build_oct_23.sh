#!/usr/bin/env bash
# Preserve the authored Beamer frames by using the TeX source as a Pandoc template.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
output_file="${1:-$script_dir/oct_23.pdf}"
if [[ "$output_file" != /* ]]; then
  output_file="$PWD/$output_file"
fi
for dependency in pandoc xelatex; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    printf 'Missing dependency: %s\n' "$dependency" >&2
    exit 1
  fi
done
mkdir -p -- "$(dirname -- "$output_file")"
cd -- "$project_dir"
pandoc --from=markdown --to=beamer --standalone \
  --template="$script_dir/oct_23.tex" --pdf-engine=xelatex \
  --output="$output_file" /dev/null
printf 'Saved %s\n' "$output_file"
