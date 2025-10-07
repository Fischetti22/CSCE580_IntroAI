
# eval.py
# Score R3 JSON files with the quiz rubric (0-100) and print results.
# Usage:
#   python eval.py data/r3_recipe1_pf1.json data/r3_recipe1_pf2.json ...

import json, sys, argparse, glob, os
from typing import Dict, Any
from r3_schema import REQUIRED_KEYS, r3_is_plausible

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_r3(path: str) -> int:
    # 50 points if valid JSON, else 0
    try:
        obj = load_json(path)
    except Exception:
        return 0

    score = 50

    # +10 each for required keys present (recipe_name, data_provenance, macronutrients, ingredients, instructions)
    for k in REQUIRED_KEYS:
        if k in obj and obj[k] is not None:
            score += 10

    # Cap at 100 by rubric
    if score > 100:
        score = 100

    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSON file paths or globs (e.g., data/*.json)")
    args = ap.parse_args()

    # Expand globs
    files = []
    for p in args.paths:
        files.extend(glob.glob(p))
    if not files:
        print("No files matched.")
        sys.exit(1)

    # Print header
    print("file,score,plausible")
    for f in sorted(files):
        try:
            obj = load_json(f)
            plausible = r3_is_plausible(obj)
        except Exception:
            plausible = False
        s = score_r3(f)
        print(f"{f},{s},{plausible}")

if __name__ == "__main__":
    main()
