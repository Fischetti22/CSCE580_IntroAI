# Quiz 2 – Directions (CSCE 580, Fall 2025)

This quiz has **two major parts (Q1 and Q2)**. Follow the steps carefully and document your answers in `docs/answer-comments.md`.

---

## Q1: Energy Consumption

**Task:** Compare the energy use of AI-based vs. traditional/simple approaches.

1. Go to the EcoByte Energy tool and run **three different settings** (e.g., Math, Writing, Generate Image).
2. Record the **AI energy consumption (Wh)**, **Simple energy consumption (Wh)**, and compute the **difference**.
3. Put results in a Markdown table with 3 rows (one per setting).
4. Compute the **average difference** across the three settings.
5. Summarize in prose: "On average, AI consumes ~X Wh more than traditional approaches."

**Deliverables (in `answer-comments.md`):**
- Table of results (3 rows)
- Average difference calculation
- 2–3 sentence commentary

---

## Q2: Recipe Conversion

**Aim:** Convert food recipes from raw text into a semi-structured JSON format (R3) using LLM prompting.

### Step 1: Select Recipes
- Pick **two recipes** from instructables.com or allrecipes.com (do not use the egg-drop soup sample).
- Save **cleaned plain-text versions** (ingredients + instructions only) into:
  - `data/original_recipe1.txt`
  - `data/original_recipe2.txt`

### Step 2: Create an AI Test Case
- Use the template provided: `docs/ai-testcase.md`
- Fill in both recipe URLs, description of cleaning, and expected JSON schema fields.

### Step 3: Run Prompt-Full (PF1–PF3)
- Use prompts PF1, PF2, PF3 (see `code/prompts_PF_PP_recipe1.txt` and `code/prompts_PF_PP_recipe2.txt`).
- For each recipe, run 3 prompts and save outputs as JSON:
  - `data/r3_recipe1_pf1.json`, `...pf2.json`, `...pf3.json`
  - `data/r3_recipe2_pf1.json`, `...pf2.json`, `...pf3.json`

### Step 4: Run Prompt-Partial (PP-1 + PP-2)
- Run PP-1 (ingredients) and PP-2 (instructions). Save as:
  - `data/ingredients_recipe1.json`, `data/instructions_recipe1.json`
  - `data/ingredients_recipe2.json`, `data/instructions_recipe2.json`
- Stitch into full R3 JSON with:
  ```bash
  python code/combine_partial.py --ingredients data/ingredients_recipe1.json --instructions data/instructions_recipe1.json --recipe_name "Recipe 1 Name" --source_url "Recipe 1 URL" --out data/r3_recipe1_pp.json

  python code/combine_partial.py --ingredients data/ingredients_recipe2.json --instructions data/instructions_recipe2.json --recipe_name "Recipe 2 Name" --source_url "Recipe 2 URL" --out data/r3_recipe2_pp.json
  ```

### Step 5: Evaluate Outputs
- Use the provided evaluator to score all JSON outputs (50 points if valid JSON +10 per required key = max 100).
  ```bash
  python code/eval.py data/*.json
  ```

### Step 6: Write Answers in `answer-comments.md`
- **Q1:** Which approach did better (PF vs. PP)?
- **Q2:** Highest score per recipe (file name + score).
- **Optional:** Document any extra prompting variations and their scores.
- **Optional:** Use GAICO to compare PF1–PF3 outputs and summarize differences.

---

## Submission Checklist

- `data/` contains 2 original text recipes + 8 JSONs (6 PF, 2 PP)
- `code/` contains all scripts (`combine_partial.py`, `eval.py`, `r3_schema.py`, `run_all.py`)
- `docs/` contains:
  - `answer-comments.md` with answers to Q1 & Q2
  - `ai-testcase.md` with recipes and schema notes
  - `quiz-directions.md` (this file)