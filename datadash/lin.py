"""
disney_predict_2032.py
======================

Train a Ridge regression model on historical Disney movie data and estimate
inflation-adjusted gross revenue for a given release year, genre(s), MPAA rating,
and (optionally) season.

The model is fit on log1p(inflation-adjusted gross) and converted back to dollars
when reporting results.

Outputs:
  - Prints cross-validation R² on the training split
  - Prints held-out test metrics (R², MAE, RMSE)
  - Prints a prediction for each requested genre for the target year
  - Saves a chart to: charts/infl_adj_gross_by_year.png

Run:
    python disney_predict_2032.py --csv disney_movie_total_gross_orchi.csv

Example:
    python disney_predict_2032.py --csv disney_movie_total_gross_orchi.csv \
        --year 2032 --genres Comedy Adventure --rating PG-13 --season Summer
"""

import os
import re
import sys
import argparse
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
GENRE_SPLIT_RE = re.compile(r"\s*(/|,|&|\+|;|\band\b|\b-\b)\s*", flags=re.IGNORECASE)

ALPHA_GRID = np.logspace(-2, 5, 60)

PALETTE = {
    "Comedy":    "#F4A261",
    "Adventure": "#2A9D8F",
    "Other":     "#AAAAAA",
}

CHART_STYLE = {
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#3A3D4A",
    "axes.labelcolor":  "#E0E0E0",
    "xtick.color":      "#AAAAAA",
    "ytick.color":      "#AAAAAA",
    "text.color":       "#E0E0E0",
    "grid.color":       "#2A2D3A",
    "grid.linewidth":   0.7,
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def fmt_money(x: float) -> str:
    x = max(0.0, float(x))
    if x >= 1e9:
        return f"${x/1e9:.3f}B"
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    return f"${x:,.0f}"


def normalize_rating(r: object) -> str:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "Unknown"
    r = str(r).strip().upper()
    return r.replace("PG 13", "PG-13").replace("PG13", "PG-13")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan}),
        errors="coerce",
    )


def parse_genres(val: object) -> List[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    parts = [p.strip() for p in GENRE_SPLIT_RE.split(s)]
    seen, out = set(), []
    for p in parts:
        if not p or p.lower() in {"/", ",", "&", "+", ";", "and", "-"}:
            continue
        g = p.title()
        if g not in seen:
            out.append(g)
            seen.add(g)
    return out


def build_genre_binary(df: pd.DataFrame, genre_col: str) -> pd.DataFrame:
    all_lists = df[genre_col].apply(parse_genres)
    unique = sorted({g for lst in all_lists for g in lst})
    mat = np.zeros((len(df), len(unique)), dtype=np.int8)
    idx = {g: i for i, g in enumerate(unique)}
    for row_i, lst in enumerate(all_lists):
        for g in lst:
            j = idx.get(g)
            if j is not None:
                mat[row_i, j] = 1
    return pd.DataFrame(mat, columns=[f"Genre__{g}" for g in unique], index=df.index)


def infer_release_year(df: pd.DataFrame) -> pd.Series:
    if "Release Year" in df.columns:
        y = pd.to_numeric(df["Release Year"], errors="coerce")
        y = y.where((y >= 1900) & (y <= 2100))
        if y.notna().sum() > len(df) * 0.5:
            return y

    if "Date Released" in df.columns:
        for dayfirst in (True, False):
            dt = pd.to_datetime(df["Date Released"], dayfirst=dayfirst, errors="coerce")
            if dt.notna().mean() > 0.8:
                y = dt.dt.year
                return y.where(y.between(1900, 2100))

    raise ValueError("Cannot find 'Release Year' or parseable 'Date Released'.")


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Modelling
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(genre_cols: List[str]) -> Pipeline:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    prep = ColumnTransformer(
        transformers=[
            ("num",   num_pipe,  ["Release_Year"]),
            ("cat",   cat_pipe,  ["MPAA_Rating_Clean", "Season_Clean"]),
            ("genre", "passthrough", genre_cols),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("prep",  prep),
        ("model", RidgeCV(alphas=ALPHA_GRID, scoring="r2")),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Charting
# ─────────────────────────────────────────────────────────────────────────────
def build_chart(
    model_df: pd.DataFrame,
    target_col: str,
    pipe: Pipeline,
    predictions: dict,           # {genre: predicted_value}
    genre_cols: List[str],
    args: argparse.Namespace,
    best_alpha: float,
    cv_r2_mean: float,
    cv_r2_std: float,
    test_r2: float,
) -> None:
    matplotlib.rcParams.update(CHART_STYLE)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Disney Movies — Inflation-Adjusted Gross\n"
        f"Forecast for {args.year}  ·  {args.rating}  ·  {args.season}",
        fontsize=15, fontweight="bold", color="#FFFFFF", y=0.98,
    )

    gs = fig.add_gridspec(
        2, 2, hspace=0.46, wspace=0.32,
        left=0.07, right=0.97, top=0.91, bottom=0.06
    )

    genre_to_plot = [g.title() for g in args.genres]

    # ── A: Yearly averages + film scatter + trend + forecast markers ─────────
    ax_main = fig.add_subplot(gs[0, :])

    yearly = (
        model_df[["Release_Year", target_col]]
        .astype({"Release_Year": int})
        .groupby("Release_Year", as_index=False)[target_col].mean()
    )

    ax_main.bar(
        yearly["Release_Year"], yearly[target_col],
        color="#264653", alpha=0.55, width=0.8, zorder=1, label="Yearly average"
    )

    for _, row in model_df.iterrows():
        g = None
        for gc in genre_cols:
            if row[gc] == 1:
                g = gc.replace("Genre__", "")
                break
        c = PALETTE.get(g, PALETTE["Other"]) if g in genre_to_plot else PALETTE["Other"]
        alpha = 0.75 if g in genre_to_plot else 0.18
        ax_main.scatter(row["Release_Year"], row[target_col], color=c, alpha=alpha, s=30, zorder=3)

    years_rng = np.arange(int(model_df["Release_Year"].min()), args.year + 1)

    for genre in genre_to_plot:
        clr = PALETTE.get(genre, "#FFFFFF")
        rows = []
        for yr in years_rng:
            r = {col: 0 for col in model_df.drop(columns=[target_col]).columns}
            r["Release_Year"] = yr
            r["MPAA_Rating_Clean"] = normalize_rating(args.rating)
            r["Season_Clean"] = args.season.title()
            gcol = f"Genre__{genre}"
            if gcol in r:
                r[gcol] = 1
            rows.append(r)

        trend_X = pd.DataFrame(rows)
        trend_log = pipe.predict(trend_X)
        trend_val = np.expm1(trend_log).clip(min=0)
        ax_main.plot(
            years_rng, trend_val, color=clr, lw=2.2, ls="--",
            alpha=0.85, zorder=4, label=f"{genre} trend"
        )

    for genre, val in predictions.items():
        clr = PALETTE.get(genre, "#FF6B6B")
        ax_main.scatter(
            args.year, val, marker="*", s=600, color=clr,
            edgecolors="white", lw=0.8, zorder=8,
            label=f"{genre} forecast: {fmt_money(val)}"
        )
        ax_main.annotate(
            f"{genre}\n{fmt_money(val)}",
            xy=(args.year, val),
            xytext=(args.year - 8, val * 1.3),
            fontsize=8.5, color=clr, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=clr, lw=1.2),
        )

    max_train_year = int(model_df["Release_Year"].max())
    if args.year > max_train_year:
        ax_main.axvspan(
            max_train_year, args.year + 1,
            alpha=0.08, color="#FF6B6B",
            label=f"Beyond last year ({max_train_year})"
        )
        ax_main.axvline(max_train_year, color="#FF6B6B", lw=1.2, ls=":", alpha=0.7)

    ax_main.set_xlabel("Release Year", fontsize=11)
    ax_main.set_ylabel("Inflation-Adjusted Gross ($)", fontsize=11)
    ax_main.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M")
    )
    ax_main.set_title(
        f"CV R²={cv_r2_mean:.3f}±{cv_r2_std:.3f}  ·  Test R²={test_r2:.3f}  ·  Best α={best_alpha:.2f}",
        fontsize=10, color="#AAAAAA",
    )
    ax_main.grid(True, ls="--", alpha=0.4, zorder=0)
    ax_main.legend(
        fontsize=8.5, loc="upper right", ncol=2,
        facecolor="#1A1D27", edgecolor="#3A3D4A", labelcolor="#E0E0E0"
    )

    # ── B: Season × Rating heatmap ───────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1, 0])
    pivot = (
        model_df
        .assign(Season=model_df["Season_Clean"], Rating=model_df["MPAA_Rating_Clean"])
        .pivot_table(values=target_col, index="Season", columns="Rating", aggfunc="mean", fill_value=0)
        / 1e6
    )
    im = ax_heat.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax_heat.set_xticks(range(len(pivot.columns)))
    ax_heat.set_xticklabels(pivot.columns, fontsize=9)
    ax_heat.set_yticks(range(len(pivot.index)))
    ax_heat.set_yticklabels(pivot.index, fontsize=9)

    vmax = pivot.values.max() if pivot.values.size else 0
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            text_color = "white" if vmax and v > vmax * 0.55 else "black"
            ax_heat.text(
                j, i, f"${v:.0f}M",
                ha="center", va="center",
                fontsize=8.5, color=text_color, fontweight="bold"
            )
    plt.colorbar(im, ax=ax_heat, label="Avg Gross ($M)", shrink=0.85)
    ax_heat.set_title("Avg Gross: Season × MPAA Rating", fontsize=11)

    # ── C: Historical vs Forecast comparison ──────────────────────────────────
    ax_cmp = fig.add_subplot(gs[1, 1])

    target_genres = [g.title() for g in args.genres]
    genre_hist = {}
    genre_rating = {}

    for g in target_genres:
        gcol = f"Genre__{g}"
        if gcol in model_df.columns:
            mask_g = model_df[gcol] == 1
            mask_r = mask_g & (model_df["MPAA_Rating_Clean"] == normalize_rating(args.rating))
            genre_hist[g] = model_df.loc[mask_g, target_col].mean() if mask_g.sum() else 0
            genre_rating[g] = model_df.loc[mask_r, target_col].mean() if mask_r.sum() else 0
        else:
            genre_hist[g] = 0
            genre_rating[g] = 0

    bar_labels, bar_values, bar_colors, bar_hatches = [], [], [], []
    for g in target_genres:
        clr = PALETTE.get(g, "#AAAAAA")
        bar_labels += [f"{g}\nAll Avg", f"{g}\n{args.rating} Avg", f"{g}\n{args.year}"]
        bar_values += [genre_hist[g], genre_rating[g], predictions.get(g, 0)]
        bar_colors += [clr, clr, clr]
        bar_hatches += ["", "..", "///"]

    for i, (lbl, val, clr, hat) in enumerate(zip(bar_labels, bar_values, bar_colors, bar_hatches)):
        alpha = 0.5 if "Avg" in lbl else 1.0
        ax_cmp.bar(i, val, color=clr, alpha=alpha, hatch=hat, edgecolor="white", lw=0.8)
        ax_cmp.text(
            i, (val * 1.015) if val > 0 else 0,
            fmt_money(val),
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#E0E0E0"
        )

    ax_cmp.set_xticks(range(len(bar_labels)))
    ax_cmp.set_xticklabels(bar_labels, fontsize=8.5)
    ax_cmp.set_ylabel("Inflation-Adjusted Gross", fontsize=10)
    ax_cmp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.0f}M"))
    ax_cmp.set_title(f"Historical Avg vs Forecast ({args.rating}, {args.season})", fontsize=11)
    ax_cmp.grid(True, axis="y", ls="--", alpha=0.35)

    legend_patches = [
        mpatches.Patch(facecolor="grey", alpha=0.5, label="All historical avg"),
        mpatches.Patch(facecolor="grey", alpha=0.5, hatch="..", label=f"{args.rating} historical avg"),
        mpatches.Patch(facecolor="grey", alpha=1.0, hatch="///", label=f"{args.year} forecast"),
    ]
    ax_cmp.legend(
        handles=legend_patches, fontsize=8, loc="upper right",
        facecolor="#1A1D27", edgecolor="#3A3D4A", labelcolor="#E0E0E0"
    )

    out_dir = os.path.dirname(args.chart_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.chart_out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {args.chart_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Disney inflation-adjusted gross using Ridge regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", default="disney movie total gross_cleaned.csv")
    parser.add_argument("--year", type=int, default=2032)
    parser.add_argument("--genres", nargs="+", default=["Comedy", "Adventure"])
    parser.add_argument("--rating", default="PG-13")
    parser.add_argument("--season", default="Summer", choices=["Spring", "Summer", "Fall", "Winter"])
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--chart_out", default="charts/infl_adj_gross_by_year.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"ERROR: CSV not found: {args.csv}")

    df_raw = pd.read_csv(args.csv)

    target_col = find_col(df_raw, ["Inflation Adjusted Gross", "Inflation-Adjusted Gross"])
    if target_col is None:
        sys.exit(f"ERROR: No target column found. Columns: {list(df_raw.columns)}")

    genre_col = find_col(df_raw, ["Genre", "Genres"])
    if genre_col is None:
        sys.exit("ERROR: No Genre column found (expected 'Genre' or 'Genres').")

    rating_col = find_col(df_raw, ["MPAA Rating", "Rating", "MPAA"])
    season_col = find_col(df_raw, ["Season"])

    df = df_raw.copy()
    df[target_col] = clean_numeric(df[target_col])
    df["Release_Year"] = infer_release_year(df)
    df["MPAA_Rating_Clean"] = df[rating_col].apply(normalize_rating) if rating_col else "Unknown"
    df["Season_Clean"] = df[season_col].astype(str).str.title() if season_col else "Unknown"

    genre_bin = build_genre_binary(df, genre_col)
    genre_cols = list(genre_bin.columns)

    model_df = pd.concat(
        [df[["Release_Year", "MPAA_Rating_Clean", "Season_Clean", target_col]], genre_bin],
        axis=1,
    ).dropna(subset=["Release_Year", target_col]).reset_index(drop=True)

    model_df[target_col] = model_df[target_col].clip(lower=0)

    if len(model_df) < 30:
        sys.exit(f"ERROR: Only {len(model_df)} usable rows (need at least 30).")

    X = model_df.drop(columns=[target_col])
    y = np.log1p(model_df[target_col].astype(float))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    pipe = build_pipeline(genre_cols)

    print(f"\nRunning {args.cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=args.cv_folds, scoring="r2", n_jobs=-1)
    cv_r2_mean, cv_r2_std = cv_scores.mean(), cv_scores.std()
    print(f"  CV R^2 : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}  (folds: {np.round(cv_scores, 3)})")

    pipe.fit(X_train, y_train)
    best_alpha = pipe.named_steps["model"].alpha_

    y_test_pred = pipe.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_test_pred).clip(0))
    test_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_test_pred).clip(0)))

    print(f"  Best alpha: {best_alpha:.4f}")
    print(f"  Test R^2  : {test_r2:.4f}")
    print(f"  Test MAE  : {fmt_money(test_mae)}")
    print(f"  Test RMSE : {fmt_money(test_rmse)}")

    pipe.fit(X, y)

    max_train_year = int(model_df["Release_Year"].max())
    min_train_year = int(model_df["Release_Year"].min())
    if args.year > max_train_year:
        print(
            f"\nWARNING: {args.year} is {args.year - max_train_year} year(s) after the last training year ({max_train_year})."
        )
    elif args.year < min_train_year:
        print(f"\nWARNING: {args.year} is before the first training year ({min_train_year}).")

    desired_genres = [g.title() for g in args.genres]
    desired_rating = normalize_rating(args.rating)
    desired_season = args.season.title()

    predictions = {}
    for genre in desired_genres:
        pred_row = {col: 0 for col in X.columns}
        pred_row["Release_Year"] = args.year
        pred_row["MPAA_Rating_Clean"] = desired_rating
        pred_row["Season_Clean"] = desired_season

        gcol = f"Genre__{genre}"
        if gcol in pred_row:
            pred_row[gcol] = 1
        else:
            print(f"  Note: genre '{genre}' was not present in the training data.")

        X_pred = pd.DataFrame([pred_row])
        log_pred = float(pipe.predict(X_pred)[0])
        val = float(np.expm1(log_pred))
        predictions[genre] = max(0.0, val)

    print("\n" + "═" * 65)
    print(f"{'  PREDICTION SUMMARY  ':^65}")
    print("═" * 65)
    print(f"  Year   : {args.year}  |  Rating: {desired_rating}  |  Season: {desired_season}")
    print(f"  Model  : RidgeCV  |  alpha={best_alpha:.2f}")
    print(f"  CV R^2 : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}  |  Test R^2: {test_r2:.4f}")
    print("─" * 65)
    for genre, val in predictions.items():
        print(f"  {genre:<14}: {fmt_money(val):>14}")
    print("═" * 65)

    print("\nBuilding chart...")
    build_chart(
        model_df=model_df,
        target_col=target_col,
        pipe=pipe,
        predictions=predictions,
        genre_cols=genre_cols,
        args=args,
        best_alpha=best_alpha,
        cv_r2_mean=cv_r2_mean,
        cv_r2_std=cv_r2_std,
        test_r2=test_r2,
    )
    print("Done.\n")


if __name__ == "__main__":
    main()