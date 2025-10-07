
# run_all.py
# Convenience: score all JSONs under data/ and print a sorted summary.

import subprocess, sys, glob, os

def main():
    files = sorted(glob.glob("data/*.json"))
    if not files:
        print("No JSON files found in data/.")
        sys.exit(1)

    cmd = [sys.executable, "code/eval.py"] + files
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
