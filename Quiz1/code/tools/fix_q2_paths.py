#!/usr/bin/env python3
import sys
import json
import re
from pathlib import Path


def decide_replacement(line: str) -> str | None:
    l = line.lower()
    # Prioritize explicit API detection
    if ".to_csv(" in l or "to_csv(" in l:
        return "data_dir"
    if "savefig(" in l or "plt.savefig(" in l:
        return "fig_dir"
    # Heuristics by extension as fallback
    if ".csv" in l or ".xlsx" in l or ".ods" in l:
        return "data_dir"
    if ".png" in l or ".jpg" in l or ".jpeg" in l or ".svg" in l:
        return "fig_dir"
    return None


def patch_notebook(nb_path: Path) -> int:
    data = json.loads(nb_path.read_text(encoding="utf-8"))

    if "cells" not in data or not isinstance(data["cells"], list):
        print(f"[!] {nb_path} does not look like a valid notebook: no cells array", file=sys.stderr)
        return 2

    total_lines = 0
    replaced = 0
    ambiguous = []

    for cell in data["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source")
        if isinstance(src, str):
            lines = src.splitlines(True)
        elif isinstance(src, list):
            lines = src
        else:
            continue

        new_lines = []
        for line in lines:
            total_lines += 1
            if "save_dir" in line:
                repl = decide_replacement(line)
                if repl is None:
                    ambiguous.append(line.rstrip("\n"))
                    new_lines.append(line)
                else:
                    # Replace raw occurrences of save_dir including inside f-strings and os.path.join
                    new_line = re.sub(r"\bsave_dir\b", repl, line)
                    if new_line != line:
                        replaced += 1
                    new_lines.append(new_line)
            else:
                new_lines.append(line)

        # write back
        if isinstance(src, str):
            cell["source"] = "".join(new_lines)
        else:
            cell["source"] = new_lines

    if replaced == 0:
        print("[i] No occurrences of save_dir found to replace.")
    else:
        print(f"[+] Replaced {replaced} line(s) that referenced save_dir.")

    if ambiguous:
        print("[!] Ambiguous lines (left unchanged). Review manually if needed:")
        for l in ambiguous[:20]:
            print("    ", l)
        if len(ambiguous) > 20:
            print(f"    ... and {len(ambiguous) - 20} more")

    backup = nb_path.with_suffix(nb_path.suffix + ".bak")
    backup.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    # Overwrite original with pretty-printed JSON to preserve readability
    nb_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Updated notebook written. Backup also written to {backup}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: fix_q2_paths.py /absolute/path/to/Q2-code.ipynb", file=sys.stderr)
        sys.exit(2)
    nbp = Path(sys.argv[1]).expanduser().resolve()
    if not nbp.exists():
        print(f"Notebook not found: {nbp}", file=sys.stderr)
        sys.exit(2)
    sys.exit(patch_notebook(nbp))