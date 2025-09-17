#!/usr/bin/env python3
# q3_fire_analysis.py

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---------- robust datetime parsing ----------
CANDIDATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %I:%M %p",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M",
]

def parse_dt_series(s: pd.Series, dayfirst=False) -> pd.Series:
    s = s.copy()
    # numeric epoch/serials
    if pd.api.types.is_numeric_dtype(s):
        a = s.astype("float64")
        out = pd.to_datetime(a, unit="ms", errors="coerce")
        bad = out.isna()
        out.loc[bad] = pd.to_datetime(a[bad], unit="s", errors="coerce")
        bad = out.isna()
        maybe_excel = (a > 20000) & (a < 60000)
        if bad.any() and maybe_excel.any():
            excel = pd.to_datetime("1899-12-30") + pd.to_timedelta(a, unit="D")
            out.loc[bad & maybe_excel] = excel[bad & maybe_excel]
        return out

    s_str = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    mask_left = s_str.notna() & (s_str != "")
    # try explicit formats
    for fmt in CANDIDATE_FORMATS:
        try:
            parsed = pd.to_datetime(s_str.where(mask_left), format=fmt, errors="coerce", dayfirst=dayfirst)
            fill = out.isna() & parsed.notna()
            out.loc[fill] = parsed.loc[fill]
            mask_left = mask_left & out.isna()
            if not mask_left.any(): break
        except Exception:
            pass
    # fallback (mixed)
    if out.isna().any():
        try:
            parsed = pd.to_datetime(s_str.where(out.isna()), errors="coerce", dayfirst=dayfirst, format="mixed")
        except TypeError:
            parsed = pd.to_datetime(s_str.where(out.isna()), errors="coerce", dayfirst=dayfirst)
        out.loc[out.isna()] = parsed
    return out

# ---------- units counting ----------
def derive_units_count(df: pd.DataFrame) -> str | None:
    cand = next((c for c in df.columns if "unit" in c.lower()), None)
    if not cand: return None
    def count_units(val):
        if pd.isna(val): return 0
        toks = re.split(r"[,\s;/]+", str(val).strip())
        toks = [t for t in toks if t and t.upper() not in {"NONE"}]
        return len(toks)
    df["units_count"] = df[cand].apply(count_units)
    return "units_count"

# ---------- plotting helpers ----------
def save_dayhour_matrix(clean: pd.DataFrame, outdir: Path) -> None:
    if "created_dt" not in clean.columns:
        # try any dt col
        dt_cols = [c for c in clean.columns if pd.api.types.is_datetime64_any_dtype(clean[c])]
        if not dt_cols: 
            print("[DayHour] no datetime columns; skipping matrix.")
            return
        use = dt_cols[0]
    else:
        use = "created_dt"
    temp = clean.copy()
    temp["dow"] = temp[use].dt.day_name()
    temp["hour"] = temp[use].dt.hour
    mat = pd.pivot_table(temp, index="hour", columns="dow", values=temp.columns[0], aggfunc="count", fill_value=0)
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    mat = mat[[c for c in order if c in mat.columns]]
    mat["Total"] = mat.sum(axis=1)
    mat.loc["Total"] = mat.sum(axis=0)
    mat.to_csv(outdir / "b4_dow_hour_matrix.csv")

    # optional quick heatmap-ish bar grid
    plt.figure(figsize=(9,6))
    mat_no_total = mat.drop(index="Total", errors="ignore").drop(columns=["Total"], errors="ignore")
    mat_no_total.plot(kind="bar", stacked=True, legend=True)
    plt.title("Incidents by Hour × Day")
    plt.tight_layout()
    plt.savefig(outdir / "day_hour_matrix.png", dpi=150)
    plt.close()

def cluster_and_profile(clean: pd.DataFrame, outdir: Path, k=3):
    # numeric features only
    Xdf = clean.select_dtypes(include=["number"]).copy()
    # drop degenerate columns
    Xdf = Xdf.drop(columns=[c for c in Xdf.columns if Xdf[c].nunique() <= 1], errors="ignore")
    Xdf = Xdf.dropna(axis=1, how="any")
    if Xdf.shape[1] == 0:
        print("[Cluster] no numeric features available.")
        return

    X = StandardScaler().fit_transform(Xdf.values)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels_k = kmeans.fit_predict(X)
    sil_k = silhouette_score(X, labels_k) if len(set(labels_k)) > 1 else np.nan

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=k)
    labels_a = agg.fit_predict(X)
    sil_a = silhouette_score(X, labels_a) if len(set(labels_a)) > 1 else np.nan

    better = "KMeans" if (sil_k >= sil_a) else "Agglomerative"
    best_labels = labels_k if better == "KMeans" else labels_a
    print(f"KMeans silhouette: {sil_k:.4f} | Agglomerative silhouette: {sil_a:.4f} | Better: {better}")

    clean["cluster_best"] = best_labels
    clean[["CaseID","cluster_best"]].to_csv(outdir / "c_clusters.csv", index=False)

    # cluster means chart for a few interpretable vars if present
    vars_for_plot = [c for c in ["units_count","resolution_min","hour"] if c in clean.columns and pd.api.types.is_numeric_dtype(clean[c])]
    if vars_for_plot:
        means = clean.groupby("cluster_best")[vars_for_plot].mean()
        ax = means.plot(kind="bar", figsize=(9,6))
        ax.set_title("Cluster Means (best method)")
        plt.tight_layout()
        plt.savefig(outdir / "q3_cluster_means.png", dpi=150)
        plt.close()

    # numeric/datetime/categorical profiles
    # numeric
    num_cols = clean.select_dtypes(include=["number"]).columns.tolist()
    if "cluster_best" in num_cols:
        num_cols.remove("cluster_best")
    if num_cols:
        prof_num = clean.groupby("cluster_best")[num_cols].agg(["mean","median","count"])
        prof_num.to_csv(outdir / "c_cluster_profile_numeric.csv")
    # datetimes
    dt_cols = [c for c in clean.columns if pd.api.types.is_datetime64_any_dtype(clean[c])]
    if dt_cols:
        prof_dt = clean.groupby("cluster_best")[dt_cols].agg(["min","max"])
        prof_dt.to_csv(outdir / "c_cluster_profile_datetimes.csv")
    # categoricals/text
    cat_cols = clean.select_dtypes(include=["object","category"]).columns.tolist()
    def top3(s: pd.Series) -> str:
        vc = s.value_counts(dropna=True).head(3)
        return "; ".join([f"{idx}:{cnt}" for idx,cnt in vc.items()])
    if cat_cols:
        prof_cat = clean.groupby("cluster_best")[cat_cols].agg(["nunique", top3])
        prof_cat.to_csv(outdir / "c_cluster_profile_categoricals.csv")

def main():
    ap = argparse.ArgumentParser(description="Q3 Fire-station data analysis & clustering")
    ap.add_argument("--csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--outdir", default="Quiz1/outputs_q3")
    ap.add_argument("--k", type=int, default=3, help="number of clusters")
    ap.add_argument("--dayfirst", action="store_true", help="treat ambiguous dates as day-first (DD/MM)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print("Shape:", df.shape)

    # (a) Data issues
    # detect likely dt columns
    likely = re.compile(r"(date|time|created|closed|opened|reported|updated|start|end)", re.I)
    date_cols = [c for c in df.columns if likely.search(c)]
    for c in date_cols:
        df[c + "_dt"] = parse_dt_series(df[c], dayfirst=args.dayfirst)

    # data range from parsed cols
    dt_cols_all = [c for c in df.columns if c.endswith("_dt")]
    if dt_cols_all:
        all_dt = pd.concat([df[c] for c in dt_cols_all], axis=0)
        print(f"Data range: {all_dt.min()} → {all_dt.max()}")
    else:
        print("No datetime columns parsed.")

    # missingness
    miss = df.isna().mean().sort_values(ascending=False).to_frame("missing_frac")
    miss.to_csv(outdir / "a2_missing_by_column.csv")

    # column summary
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "example": df.apply(lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else None)
    })
    summary.to_csv(outdir / "a3_column_summary.csv")

    # (a.4) Cleaning
    clean = df.copy()
    for c in clean.columns:
        if clean[c].dtype == object:
            clean[c] = clean[c].astype(str).str.strip()
    # canonical created/closed if present
    created_col = next((c for c in clean.columns if c.lower().startswith("created")), None)
    closed_col  = next((c for c in clean.columns if c.lower().startswith("closed")), None)
    if created_col and f"{created_col}_dt" in clean.columns:
        clean["created_dt"] = clean[f"{created_col}_dt"]
    if closed_col and f"{closed_col}_dt" in clean.columns:
        clean["closed_dt"] = clean[f"{closed_col}_dt"]

    # resolution minutes
    if "created_dt" in clean and "closed_dt" in clean:
        clean["resolution_min"] = (pd.to_datetime(clean["closed_dt"]) - pd.to_datetime(clean["created_dt"])).dt.total_seconds() / 60.0

    # add CaseID
    clean["CaseID"] = np.arange(1, len(clean) + 1)

    # impute missing
    for c in clean.columns:
        if pd.api.types.is_numeric_dtype(clean[c]):
            med = pd.to_numeric(clean[c], errors="coerce").median()
            clean[c] = pd.to_numeric(clean[c], errors="coerce").fillna(med)
        elif pd.api.types.is_datetime64_any_dtype(clean[c]):
            # leave datetimes as NaT
            pass
        else:
            clean[c] = clean[c].replace({"": np.nan}).fillna("Unknown")

    # (b) Exploratory stats
    avg_res = clean["resolution_min"].mean() if "resolution_min" in clean else np.nan
    print("Average resolution (min):", round(avg_res,2) if pd.notna(avg_res) else "N/A")

    units_name = derive_units_count(clean)  # creates 'units_count' if possible
    avg_units = clean[units_name].mean() if units_name else np.nan
    print("Average units per alarm:", round(avg_units,2) if pd.notna(avg_units) else "N/A (no units column)")

    shift_col = next((c for c in clean.columns if "shift" in c.lower()), None)
    if shift_col:
        counts = clean[shift_col].value_counts()
        busiest = counts.idxmax()
        print("Busiest shift:", busiest, "| counts:", dict(counts))
    else:
        print("No shift column detected.")

    save_dayhour_matrix(clean, outdir)

    # (c) Clustering
    cluster_and_profile(clean, outdir, k=args.k)

    # (optional) save cleaned data
    clean.to_csv(outdir / "cleaned_data.csv", index=False)
    print("[Done] Q3 artifacts saved in:", outdir)

if __name__ == "__main__":
    main()

