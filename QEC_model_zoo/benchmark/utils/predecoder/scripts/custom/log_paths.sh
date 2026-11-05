# shellcheck shell=bash
# Keep file paths in pre-decoder logs relative: absolute paths under the repository root
# become repo-relative (e.g. QEC_model_zoo/benchmark/...), and other paths under $HOME
# start with ~/. Sourced by the runners in scripts/custom and scripts/nvidia/local_run.sh.

# This file lives at <repo>/QEC_model_zoo/benchmark/utils/predecoder/scripts/custom/.
LOG_PATHS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"

# Stream filter: relpath_filter < in > out (flushes every line, safe before tee).
relpath_filter() {
  awk -v root="${LOG_PATHS_ROOT}" -v home="${HOME:-}" '
    function replace_all(s, old, new,    i, out) {
      out = ""
      while ((i = index(s, old)) > 0) {
        out = out substr(s, 1, i - 1) new
        s = substr(s, i + length(old))
      }
      return out s
    }
    {
      line = replace_all($0, root "/", "")
      line = replace_all(line, root, ".")
      if (home != "" && home != "/") line = replace_all(line, home "/", "~/")
      print line
      fflush()
    }'
}

# Rewrite every *.log file under a directory in place (e.g. Hydra job logs written by Python).
relpath_logs() {
  [ -d "$1" ] || return 0
  find "$1" -type f -name "*.log" | while IFS= read -r f; do
    tmp="$(mktemp "${f}.XXXXXX")"
    relpath_filter < "$f" > "$tmp" && cat "$tmp" > "$f"
    rm -f "$tmp"
  done
}
