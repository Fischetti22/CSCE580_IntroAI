# Quiz 2 - Answer Comments

## Q1: Energy Consumption

We tested 3 different settings with the EcoByte AI vs Simple Energy tool:

| Setting | Action            | AI Energy | Simple Energy | Difference (AI − Simple) |
|---------|------------------|-----------|---------------|--------------------------|
| 1       | Math (adjusted)  | 0.70 Wh   | 0.01 Wh       | 0.69 Wh                  |
| 2       | Writing (default)| 1.50 Wh   | 0.10 Wh       | 1.40 Wh                  |
| 3       | Generate Image   | 2.90 Wh   | 0.30 Wh       | 2.60 Wh                  |

**Average difference** = (0.69 + 1.40 + 2.60) / 3 = **1.56 Wh**

On average, AI consumes more energy than traditional alternatives by ~1.6 Wh.

---

## Q2: Using LLMs for recipe conversion

Artifacts (all under Quiz-2/):
- Data: data/original_recipe1.txt, data/original_recipe2.txt
- PF outputs (full R3):
  - data/r3_recipe1_pf1.json, pf2.json, pf3.json
  - data/r3_recipe2_pf1.json, pf2.json, pf3.json
- PP outputs (partial + stitched):
  - data/ingredients_recipe1.json, data/instructions_recipe1.json -> data/r3_recipe1_pp.json
  - data/ingredients_recipe2.json, data/instructions_recipe2.json -> data/r3_recipe2_pp.json
- Prompts: data/prompts_PF_PP_recipe1.txt, data/prompts_PF_PP_recipe2.txt
- Scoring script: code/eval.py (uses code/r3_schema.py)

Evaluation results (python Quiz-2/code/eval.py Quiz-2/data/r3_*.json):
- r3_recipe1_pf1.json = 100
- r3_recipe1_pf2.json = 100
- r3_recipe1_pf3.json = 100
- r3_recipe1_pp.json  = 100
- r3_recipe2_pf1.json = 100
- r3_recipe2_pf2.json = 100
- r3_recipe2_pf3.json = 100
- r3_recipe2_pp.json  = 100

Answers:
- Q1 (Which approach did better?): Under this rubric, PF and PP performed equally well (all scored 100). Qualitatively, PP offers more control over structure when LLMs tend to drift, while PF is faster when the schema is followed reliably.
- Q2 (Highest goodness score per food?):
  - Blackened Tofu and Vegetables: 100
  - Rhubarb Crumble (Vegan): 100

## Extra Credit
- Alternate prompting strategy (schema-first/repair):
  - Files: data/r3_recipe1_alt.json, data/r3_recipe2_alt.json
  - Scores: r3_recipe1_alt.json = 100; r3_recipe2_alt.json = 100
  - Observation: Schema-first prompts made it straightforward to enforce required keys and atomic steps; results matched PF/PP scores but with fewer retries.
- GAICO comparison (PF1 vs PF2 vs PF3) for Rhubarb Crumble (Vegan):
  - All three PF files share nearly identical ingredient sets and one-action steps; minimal variance in step phrasing (e.g., cut vs chop).
  - Structure is consistent across PF variants, suggesting stable extraction under the given schema and constraints.
  - Any differences are cosmetic and do not affect rubric scoring.
