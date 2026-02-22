# disney_predict_2032.py
#
# Predict Inflation Adjusted Gross from historical Disney data.
# Uses Ridge regression on log1p(Inflation Adjusted Gross) and converts back.
#
# Run:
#   python disney_predict_2032.py
#   python disney_predict_2032.py --year 2032 --genres Comedy Adventure --rating PG-13
#
# Output:
#   Prints prediction
#   Saves chart to: charts/infl_adj_gross_by_year.png

import os
import re
import sys
import argparse
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

GENRE_SPLIT_RE = re.compile(r"\s*(/|,|&|\+|;|\band\b|\b-\b)\s*", flags=re.IGNORECASE)


def money(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    x = max(0.0, float(x))
    return f"${x:,.0f}"


def normalize_rating(r: object) -> str:
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return ""
    r = str(r).strip().upper()
    r = r.replace("PG 13", "PG-13").replace("PG13", "PG-13")
    return r


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
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

    genres: List[str] = []
    for p in parts:
        p = p.strip()
        if not p or p.lower() in ["/", ",", "&", "+", ";", "and", "-"]:
            continue
        genres.append(p.title())

    seen = set()
    out = []
    for g in genres:
        if g not in seen:
            out.append(g)
            seen.add(g)
    return out


def build_multigenre_frame(df: pd.DataFrame, genre_col: str) -> pd.DataFrame:
    all_lists = df[genre_col].apply(parse_genres)

    unique = set()
    for lst in all_lists:
        unique.update(lst)
    unique = sorted(unique)

    mat = np.zeros((len(df), len(unique)), dtype=int)
    genre_to_idx = {g: i for i, g in enumerate(unique)}

    for row_i, lst in enumerate(all_lists):
        for g in lst:
            j = genre_to_idx.get(g)
            if j is not None:
                mat[row_i, j] = 1

    return pd.DataFrame(mat, columns=[f"Genre__{g}" for g in unique], index=df.index)


def infer_release_year(df: pd.DataFrame) -> pd.Series:
    if "Release Year" in df.columns:
        y = pd.to_numeric(df["Release Year"], errors="coerce")
        y = y.where((y >= 1900) & (y <= 2100), np.nan)
        if y.notna().sum() > 0:
            return y

    if "Date Released" in df.columns:
        dt = pd.to_datetime(df["Date Released"], dayfirst=True, errors="coerce")
        if dt.isna().mean() > 0.5:
            dt2 = pd.to_datetime(df["Date Released"], dayfirst=False, errors="coerce")
            if dt2.notna().sum() > dt.notna().sum():
                dt = dt2

        y = dt.dt.year
        y = y.where((y >= 1900) & (y <= 2100), np.nan)
        return y

    raise ValueError("Could not infer release year from 'Release Year' or 'Date Released'.")


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Predict Inflation Adjusted Gross using Ridge regression."
    )
    parser.add_argument("--csv", type=str, default="disney movie total gross.csv", help="Path to CSV file")
    parser.add_argument("--year", type=int, default=2032, help="Release year")
    parser.add_argument("--genres", nargs="+", default=["Comedy", "Adventure"], help="Genres")
    parser.add_argument("--rating", type=str, default="PG-13", help="MPAA rating (e.g., PG-13)")
    parser.add_argument("--alpha", type=float, default=3.0, help="Ridge alpha")
    parser.add_argument(
        "--chart_out",
        type=str,
        default="charts/infl_adj_gross_by_year.png",
        help="Output chart path",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found at: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df_raw = pd.read_csv(args.csv)

    target_col = find_first_existing_column(df_raw, ["Inflation Adjusted Gross", "Inflation-Adjusted Gross"])
    if target_col is None:
        print("ERROR: Missing 'Inflation Adjusted Gross' column.", file=sys.stderr)
        print(f"Columns found: {list(df_raw.columns)}", file=sys.stderr)
        sys.exit(1)

    genre_col = find_first_existing_column(df_raw, ["Genre", "Genres"])
    if genre_col is None:
        print("ERROR: Missing Genre column (expected 'Genre' or 'Genres').", file=sys.stderr)
        sys.exit(1)

    rating_col = find_first_existing_column(df_raw, ["MPAA Rating", "Rating", "MPAA"])

    df = df_raw.copy()
    df[target_col] = clean_numeric(df[target_col])
    df["Release_Year"] = infer_release_year(df)

    if rating_col is not None:
        df["MPAA_Rating_Clean"] = df[rating_col].apply(normalize_rating)
    else:
        df["MPAA_Rating_Clean"] = ""

    genre_bin = build_multigenre_frame(df, genre_col)

    model_df = pd.concat(
        [
            df[["Release_Year", "MPAA_Rating_Clean", target_col]],
            genre_bin,
        ],
        axis=1,
    )

    model_df = model_df.dropna(subset=["Release_Year", target_col]).reset_index(drop=True)

    if len(model_df) < 30:
        print(f"ERROR: Not enough rows to train (got {len(model_df)}).", file=sys.stderr)
        sys.exit(1)

    model_df[target_col] = model_df[target_col].clip(lower=0)

    X = model_df.drop(columns=[target_col])
    y = model_df[target_col].astype(float)

    y_log = np.log1p(y)

    numeric_features = ["Release_Year"]
    categorical_features = ["MPAA_Rating_Clean"]
    genre_features = [c for c in X.columns if c.startswith("Genre__")]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
            ("genre", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), genre_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", Ridge(alpha=float(args.alpha))),
    ])

    pipe.fit(X, y_log)

    desired_genres = [str(g).title() for g in args.genres]
    desired_rating = normalize_rating(args.rating)

    pred_row = {col: 0 for col in X.columns}
    pred_row["Release_Year"] = int(args.year)
    pred_row["MPAA_Rating_Clean"] = desired_rating

    for g in desired_genres:
        col = f"Genre__{g}"
        if col in pred_row:
            pred_row[col] = 1

    X_pred = pd.DataFrame([pred_row])

    y_pred_log = float(pipe.predict(X_pred)[0])
    y_pred = float(np.expm1(y_pred_log))
    y_pred = max(0.0, y_pred)

    print("\n=== Prediction ===")
    print(f"Release year: {args.year}")
    print(f"Genres: {', '.join(desired_genres)}")
    print(f"Rating: {desired_rating}")
    print(f"Predicted Inflation Adjusted Gross: {money(y_pred)}")

    chart_df = model_df[["Release_Year", target_col]].copy()
    chart_df["Release_Year"] = chart_df["Release_Year"].astype(int)
    yearly = chart_df.groupby("Release_Year", as_index=False)[target_col].mean()

    out_dir = os.path.dirname(args.chart_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.bar(yearly["Release_Year"], yearly[target_col])
    plt.xlabel("Release Year")
    plt.ylabel("Inflation Adjusted Gross ($)")
    plt.title("Inflation Adjusted Gross by Year (Yearly Average)")
    plt.tight_layout()
    plt.savefig(args.chart_out, dpi=200)
    plt.close()

    print(f"Saved chart to: {args.chart_out}\n")


if __name__ == "__main__":
    main()