
# combine_partial.py
# Stitch partial outputs (ingredients + instructions) into a full R3 JSON.
# Usage:
#   python combine_partial.py --ingredients data/ingredients_recipe1.json --instructions data/instructions_recipe1.json \#       --recipe_name "Blackened Tofu and Vegetables" --source_url "https://..." --out data/r3_recipe1_pp.json

import json, argparse, datetime
from typing import Any, Dict, List

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingredients", required=True, help="Path to ingredients JSON array")
    ap.add_argument("--instructions", required=True, help="Path to instructions JSON array")
    ap.add_argument("--recipe_name", required=True)
    ap.add_argument("--source_url", required=True)
    ap.add_argument("--out", required=True, help="Output R3 JSON path")
    args = ap.parse_args()

    ingredients = load_json(args.ingredients)
    instructions = load_json(args.instructions)

    r3 = {
        "recipe_name": args.recipe_name,
        "data_provenance": {
            "source_url": args.source_url,
            "retrieved_at": datetime.datetime.utcnow().isoformat() + "Z",
        },
        "macronutrients": {
            "calories": None, "protein_g": None, "fat_g": None, "carbs_g": None
        },
        "ingredients": ingredients,
        "instructions": instructions
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(r3, f, ensure_ascii=False, indent=2)

    print(f"[combine_partial] Wrote {args.out}")

if __name__ == "__main__":
    main()
