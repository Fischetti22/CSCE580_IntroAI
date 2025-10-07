# AI Testcase: Recipe text to R3 JSON (Two foods, PF and PP approaches)

- Testcase ID: Q2-R3-Conversion-BlackenedTofu-RhubarbCrumble
- Author: Pedro Fischetti
- Date: 2025-10-07
- Purpose: Convert two web recipes from natural language to R3 JSON suitable for a robotic chef. Compare Prompt-Full (PF) vs Prompt-Partial (PP) conversion workflows.

## Inputs
- Food 1 URL: https://www.instructables.com/Blackened-Tofu-and-Vegetables/
- Food 2 URL: https://www.instructables.com/Rhubarb-Crumble-Vegan/
- Source texts saved to:
  - Quiz-2/data/original_recipe1.txt
  - Quiz-2/data/original_recipe2.txt

## Models/Tools
- LLM: ChatGPT (gpt-5) used to author conversion prompts/outputs.
- Local scripts:
  - Quiz-2/code/combine_partial.py to stitch ingredients+instructions into R3 (PP approach)
  - Quiz-2/code/eval.py to score goodness of outputs

## Schema (R3)
- recipe_name: string
- data_provenance: { source_url: string, retrieved_at: ISO8601 string }
- macronutrients: { calories: number|null, protein_g: number|null, fat_g: number|null, carbs_g: number|null }
- ingredients: [ { item: string, quantity: number|null, unit: string|null, notes: string|null } ]
- instructions: [ { step_no: number, action: string, parameters: string|null, duration: string|null, temperature: string|null } ]

## Evaluation rubric (Goodness score out of 100)
- JSON parses: 50 points (0 if invalid JSON)
- +10 points each for presence of the following keys (max +50):
  - recipe_name
  - data_provenance
  - macronutrients
  - ingredients
  - instructions

Evaluator: Quiz-2/code/eval.py implements this rubric exactly.

## Procedures

### PF (Prompt-Full) approach
- For each food, run three distinct prompts (PF1, PF2, PF3) that request a complete R3 file in one shot.
- Prompts stored in:
  - Quiz-2/data/prompts_PF_PP_recipe1.txt (PF1–PF3 prompts for recipe1)
  - Quiz-2/data/prompts_PF_PP_recipe2.txt (PF1–PF3 prompts for recipe2)
- Expected outputs:
  - Quiz-2/data/r3_recipe1_pf1.json
  - Quiz-2/data/r3_recipe1_pf2.json
  - Quiz-2/data/r3_recipe1_pf3.json
  - Quiz-2/data/r3_recipe2_pf1.json
  - Quiz-2/data/r3_recipe2_pf2.json
  - Quiz-2/data/r3_recipe2_pf3.json

### PP (Prompt-Partial) approach
- For each food, extract partial content via two prompts:
  - PP-1: Ingredients as JSON array only
  - PP-2: Instructions as JSON array only
- Then stitch into R3 JSON using combine_partial.py with appropriate recipe_name and source_url.
- Partial files example for recipe1:
  - Quiz-2/data/ingredients_recipe1.json
  - Quiz-2/data/instructions_recipe1.json
  - Stitched: Quiz-2/data/r3_recipe1_pp.json
- Partial files example for recipe2:
  - Quiz-2/data/ingredients_recipe2.json
  - Quiz-2/data/instructions_recipe2.json
  - Stitched: Quiz-2/data/r3_recipe2_pp.json

## Success criteria
- At least 8 JSON outputs (PF1–PF3 and PP for two foods)
- Each output scores > 50 (valid JSON) under eval.py
- Documentation updated in Quiz-2/docs/answer-comments.md with answers:
  - Which approach did better (PF vs PP)?
  - Highest score for each food

## Observations
- One-action-per-step improves clarity of instructions for robotic execution.
- Unknown macronutrients are set to null.
