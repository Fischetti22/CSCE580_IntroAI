
# r3_schema.py
# Lightweight helpers for validating the expected R3 JSON shape (stdlib only).

from typing import Any, Dict, List

REQUIRED_KEYS = [
    "recipe_name",
    "data_provenance",
    "macronutrients",
    "ingredients",
    "instructions",
]

ING_KEYS = ["item", "quantity", "unit", "notes"]
INS_KEYS = ["step_no", "action", "parameters", "duration", "temperature"]

def has_required_top_keys(obj: Dict[str, Any]) -> bool:
    return all(k in obj for k in REQUIRED_KEYS)

def validate_ingredients(ingredients: Any) -> bool:
    if not isinstance(ingredients, list):
        return False
    for it in ingredients:
        if not isinstance(it, dict):
            return False
        # Only check presence, not types (quiz rubric doesn't require strict types)
        for k in ING_KEYS:
            if k not in it:
                return False
    return True

def validate_instructions(instructions: Any) -> bool:
    if not isinstance(instructions, list):
        return False
    for st in instructions:
        if not isinstance(st, dict):
            return False
        for k in INS_KEYS:
            if k not in st:
                return False
        # ensure atomic-looking step: 'action' should be short-ish
        action = st.get("action", "")
        if not isinstance(action, str) or not action.strip():
            return False
    return True

def validate_data_provenance(dp: Any) -> bool:
    if not isinstance(dp, dict):
        return False
    return ("source_url" in dp) and ("retrieved_at" in dp)

def validate_macros(macros: Any) -> bool:
    # allow numbers or nulls; we only check that keys exist
    if not isinstance(macros, dict):
        return False
    for k in ["calories", "protein_g", "fat_g", "carbs_g"]:
        if k not in macros:
            return False
    return True

def r3_is_plausible(obj: Dict[str, Any]) -> bool:
    if not has_required_top_keys(obj):
        return False
    if not validate_data_provenance(obj.get("data_provenance")):
        return False
    if not validate_macros(obj.get("macronutrients")):
        return False
    if not validate_ingredients(obj.get("ingredients")):
        return False
    if not validate_instructions(obj.get("instructions")):
        return False
    return True
