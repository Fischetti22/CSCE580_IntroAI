REQUIRED_KEYS = ["recipe_name", "data_provenance", "macronutrients", "ingredients", "instructions"]


def r3_is_plausible(obj) -> bool:
    try:
        # Basic type checks
        if not isinstance(obj, dict):
            return False
        if not all(k in obj for k in REQUIRED_KEYS):
            return False
        dp = obj.get("data_provenance", {})
        if not isinstance(dp, dict) or "source_url" not in dp or "retrieved_at" not in dp:
            return False
        macro = obj.get("macronutrients", {})
        if not isinstance(macro, dict) or not all(k in macro for k in ["calories", "protein_g", "fat_g", "carbs_g"]):
            return False
        ings = obj.get("ingredients", [])
        if not isinstance(ings, list):
            return False
        for it in ings:
            if not isinstance(it, dict):
                return False
            # required fields in ingredient objects
            if not all(k in it for k in ["item", "quantity", "unit", "notes"]):
                return False
        instr = obj.get("instructions", [])
        if not isinstance(instr, list):
            return False
        for st in instr:
            if not isinstance(st, dict):
                return False
            if not all(k in st for k in ["step_no", "action", "parameters", "duration", "temperature"]):
                return False
        return True
    except Exception:
        return False
