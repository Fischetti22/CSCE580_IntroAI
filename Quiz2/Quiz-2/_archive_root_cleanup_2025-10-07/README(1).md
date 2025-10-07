# Quiz-2 Scaffold (CSCE 580, Fall 2025)

Folders:
- `data/` — original recipe text files + all generated R3 JSONs
- `code/` — evaluation + stitching scripts
- `docs/answer-comments.md` — written answers for Q1 & Q2 (comments, cleaning, conclusions)

Start here:
1) Put raw recipe text in `data/original_recipe1.txt` and `data/original_recipe2.txt`.
2) Run Prompt-Full (PF1–PF3) and save `data/r3_recipe{n}_pf{m}.json` (total 6 files).
3) Run Prompt-Partial (PP-1/PP-2), then stitch with `code/combine_partial.py` into `data/r3_recipe{n}_pp.json` (2 files).
4) Evaluate all 8 JSONs with `code/eval.py` -> prints goodness score (0–100).
5) Answer Q2 in `docs/answer-comments.md`.

## Extra Credit How-To

1) Alternate Prompt (schema-first / repair)
   - Open `code/prompts_alt.txt`
   - Paste Alt-1 into your LLM with the recipe text; save output to `data/r3_recipe1_alt.json` (and repeat for recipe 2).
   - Score:
     ```bash
     python code/eval.py data/r3_recipe1_alt.json data/r3_recipe2_alt.json
     ```

2) GAICO Comparison
   - Open the class GAICO notebook: `sample-code/class2-gaico-usage/LLMsOutputAnalysisWithGaico.ipynb`
   - Compare PF1/PF2/PF3 outputs for one recipe (e.g., recipe 2).
   - Summarize your findings in `docs/answer-comments.md` under **Extra Credit**.

3) Export scores to CSV (optional for your write-up)
   ```bash
   python code/scores_to_csv.py
   # Creates docs/scores.csv
   ```
