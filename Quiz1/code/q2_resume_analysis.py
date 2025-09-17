#!/usr/bin/env python3
# q2_resume_analysis.py

import argparse, re, shutil, subprocess
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# ---------- I/O helpers ----------
def has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def read_pdf_pdftotext(path: Path) -> str:
    if not has_cmd("pdftotext"):
        return ""
    tmp = path.with_suffix(".tmp.txt")
    try:
        subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(path), str(tmp)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        s = tmp.read_text(encoding="utf-8", errors="ignore")
        try: tmp.unlink()
        except Exception: pass
        return s
    except Exception:
        return ""

def read_pdf_pdfplumber(path: Path) -> str:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception:
        return ""

def read_docx(path: Path) -> str:
    try:
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    except Exception:
        return ""

def read_any(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            # try plumber → pdftotext
            t = read_pdf_pdfplumber(path)
            if not t.strip():
                t = read_pdf_pdftotext(path)
            return t
        elif ext == ".docx":
            return read_docx(path)
        elif ext == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        else:
            return ""
    except Exception:
        return ""

# ---------- text utils ----------
WORD_RE = re.compile(r"[A-Za-z][A-Za-z+'-]*")

def tokenize(text: str):
    return WORD_RE.findall(text.lower())

DEFAULT_STOPWORDS = set('''i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves
he him his himself she she's her hers herself it it's its itself they them their theirs themselves
what which who whom this that that'll these those am is are was were be been being have has had having
do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don should now and
'''.split())

def load_stopwords_file(path: Path) -> set:
    if not path: return set()
    if not path.exists(): return set()
    toks = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip().lower()
        if not ln or ln.startswith("#"): continue
        toks.extend(re.split(r"[,\s]+", ln))
    return {t for t in toks if t}

def top_n(words, n=20) -> pd.DataFrame:
    c = Counter(words)
    return pd.DataFrame(c.most_common(n), columns=["word","count"])

def barplot(df: pd.DataFrame, title: str, outpng: Path):
    plt.figure(figsize=(8,6))
    plt.bar(df["word"], df["count"])
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    outpng.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpng, dpi=150)
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Q2 Resume Analysis (my vs class)")
    ap.add_argument("--my-resume", required=True, help="Path to your resume file (pdf/docx/txt)")
    ap.add_argument("--class-folder", required=True, help="Folder with class resumes (recursively scans)")
    ap.add_argument("--outdir", default="Quiz1/outputs_q2")
    ap.add_argument("--stopwords-file", default=None, help="Optional stopwords txt file")
    ap.add_argument("--add-stopwords", default="", help="Comma-separated extra domain stopwords")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # stopwords
    STOP = set(DEFAULT_STOPWORDS)
    STOP |= load_stopwords_file(Path(args.stopwords_file)) if args.stopwords_file else set()
    if args.add_stopwords:
        STOP |= {w.strip().lower() for w in args.add_stopwords.split(",") if w.strip()}

    # ---- Task 1: my resume ----
    my_text = read_any(Path(args.my_resume))
    if not my_text.strip():
        raise SystemExit(f"[!] Could not read: {args.my_resume}")

    my_words = tokenize(my_text)
    my_top20_raw = top_n(my_words, 20)
    my_top20_raw.to_csv(outdir / "my_top20_raw.csv", index=False)
    barplot(my_top20_raw, "Task 1a: Top 20 words (raw)", outdir / "task1a_my_top20_raw.png")

    my_specific = [w for w in my_words if w not in STOP]
    my_top20_specific = top_n(my_specific, 20)
    my_top20_specific.to_csv(outdir / "my_top20_specific.csv", index=False)
    barplot(my_top20_specific, "Task 1b: Top 20 words (stopwords removed)", outdir / "task1b_my_top20_specific.png")

    # ---- Task 2: class folder ----
    folder = Path(args.class_folder).expanduser().resolve()
    if not folder.exists():
        raise SystemExit(f"[!] class folder not found: {folder}")

    class_rows = []
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".pdf",".docx",".txt"}:
            txt = read_any(f)
            if txt and txt.strip():
                class_rows.append({"filename": str(f), "text": txt})
    if not class_rows:
        raise SystemExit("[!] No readable resumes found under class folder (try OCR or convert to docx/txt).")

    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(outdir / "class_texts.csv", index=False)

    all_words = []
    for t in class_df["text"]:
        all_words.extend(tokenize(t))

    class_top20_raw = top_n(all_words, 20)
    class_top20_raw.to_csv(outdir / "class_top20_raw.csv", index=False)
    barplot(class_top20_raw, "Task 2: Class top 20 words (raw)", outdir / "task2_class_top20_raw.png")

    class_specific = [w for w in all_words if w not in STOP]
    class_top20_specific = top_n(class_specific, 20)
    class_top20_specific.to_csv(outdir / "class_top20_specific.csv", index=False)
    barplot(class_top20_specific, "Task 2: Class top 20 words (stopwords removed)", outdir / "task2_class_top20_specific.png")

    # ---- Task 3: unique to me ----
    uniq = sorted(set(my_specific) - set(class_specific))
    pd.Series(uniq, name="unique_words").to_csv(outdir / "task3_unique_words.csv", index=False)

    print("[Done] Q2 artifacts saved in:", outdir)

if __name__ == "__main__":
    main()

