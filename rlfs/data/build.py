# rlfs/data/build.py
from __future__ import annotations

import pandas as pd
import numpy as np

from .io import load_binary_csv, load_protein_data


def build_single_disease_table(
    df_patient: pd.DataFrame,
    df_protein: pd.DataFrame,
    code: str,
    sample_n: int = 3000,
):
    disease_cols = df_patient.columns[1:]
    coding_map = {}

    for c in disease_cols:
        parts = c.split("#")
        if len(parts) >= 2:
            coding_map[parts[1].strip()] = c

    if code not in coding_map:
        raise ValueError(f"Disease code {code} not found in binary table")

    col = coding_map[code]
    print(f"Target disease column: {col}")

    def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["userID"] = (
            df["userID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        return df

    df_patient = normalize_ids(df_patient)
    df_protein = normalize_ids(df_protein)

    if (
        df_patient["userID"].str.startswith("ID").any()
        and not df_protein["userID"].str.startswith("ID").any()
    ):
        df_protein["userID"] = "ID" + df_protein["userID"]

    sub = (
        df_patient.loc[~df_patient[col].isna(), ["userID", col]]
        .rename(columns={col: "label"})
    )

    merged = pd.merge(sub, df_protein, on="userID", how="inner")

    if len(merged) > sample_n:
        merged = merged.sample(n=sample_n, random_state=0)

    feat_cols = [c for c in df_protein.columns if c != "userID"]
    merged[feat_cols] = merged[feat_cols].astype(float)
    merged[feat_cols] = merged[feat_cols].fillna(
        merged[feat_cols].mean(numeric_only=True)
    )

    X = merged[feat_cols].values
    y = merged["label"].astype(int).values

    print(f"{code}: {X.shape[0]} patients × {X.shape[1]} proteins")
    return X, y, feat_cols


def build_training_data(
    disease_path: str,
    protein_paths: list[str],
    target_code: str,
    sample_n: int = 3000,
):
    print("Loading raw data...")
    df_patient = load_binary_csv(disease_path)
    df_protein = load_protein_data(protein_paths)

    return build_single_disease_table(
        df_patient=df_patient,
        df_protein=df_protein,
        code=target_code,
        sample_n=sample_n,
    )
