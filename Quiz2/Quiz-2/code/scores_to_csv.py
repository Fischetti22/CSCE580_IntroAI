# scores_to_csv.py
# Run eval.py and save results to docs/scores.csv

import csv, subprocess, sys, glob, os, json

def run_eval():
    # Collect all JSON files under data/
    files = sorted(glob.glob("data/*.json"))
    if not files:
        print("No JSON files in data/")
        sys.exit(1)
    # Run eval.py and capture stdout
    import subprocess
    proc = subprocess.run([sys.executable, "code/eval.py"] + files, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        sys.exit(proc.returncode)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    # First line is header from eval.py
    header = lines[0].split(",")
    rows = [ln.split(",") for ln in lines[1:]]
    # Write CSV
    os.makedirs("docs", exist_ok=True)
    with open("docs/scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("Wrote docs/scores.csv")

if __name__ == "__main__":
    run_eval()