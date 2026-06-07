import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_binary_csv(path: str) -> pd.DataFrame:
    print(f"Loading disease binary table: {path}")
    return pd.read_csv(path, compression="gzip")

def load_protein_data(paths):
    print("Loading Olink protein data blocks...")
    dfs, header_cols = [], None
    for i, p in enumerate(paths):
        print(f"Reading {p} ...")
        if i == 0:
            df_part = pd.read_csv(p, compression="gzip")
            if "userID" not in df_part.columns:
                df_part.rename(columns={df_part.columns[0]: "userID"}, inplace=True)
            header_cols = df_part.columns
            print(f"✅ Header detected: {len(header_cols)} columns")
        else:
            df_part = pd.read_csv(p, compression="gzip", header=None)
            df_part.columns = header_cols
        if df_part.columns[0] != "userID":
            df_part.rename(columns={df_part.columns[0]: "userID"}, inplace=True)
        for col in df_part.columns:
            if col != "userID":
                df_part[col] = pd.to_numeric(df_part[col], errors="coerce")
        dfs.append(df_part)
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]
    bad_cols = [c for c in df_all.columns if c.startswith("Unnamed") or c.replace(".", "", 1).isdigit()]
    if bad_cols:
        print(f"⚠️ Removing {len(bad_cols)} invalid columns")
        df_all = df_all.drop(columns=bad_cols)
    print(f"✅ Loaded protein data: {df_all.shape[0]} patients × {df_all.shape[1]} features")
    return df_all


def build_single_disease_table(df_patient, df_protein, code, sample_n=3000):
    disease_cols = df_patient.columns[1:]
    coding_map = {}
    for c in disease_cols:
        parts = c.split("#")
        if len(parts) >= 2:
            coding_map[parts[1].strip()] = c

    if code not in coding_map:
        raise ValueError(f"⚠️ Disease code {code} not found in binary_csv.gz")

    col = coding_map[code]
    print(f"\nTarget disease column: {col}")

    # Normalize ID format
    def norm_ids(df):
        df["userID"] = df["userID"].astype(str).str.strip()
        df["userID"] = df["userID"].str.replace(r"\.0$", "", regex=True)
        return df

    df_patient = norm_ids(df_patient)
    df_protein = norm_ids(df_protein)

    if df_patient["userID"].str.startswith("ID").any() and not df_protein["userID"].str.startswith("ID").any():
        df_protein["userID"] = "ID" + df_protein["userID"].astype(str)

    # Merge with protein table
    sub = df_patient.loc[~df_patient[col].isna(), ["userID", col]].rename(columns={col: "label"})
    merged = pd.merge(sub, df_protein, on="userID", how="inner")

    if len(merged) > sample_n:
        merged = merged.sample(n=sample_n)

    feat_cols = [c for c in df_protein.columns if c != "userID"]
    merged[feat_cols] = merged[feat_cols].astype(float).fillna(merged[feat_cols].mean(numeric_only=True))

    X = merged[feat_cols].values
    y = merged["label"].astype(int).values
    print(f"✅ {code}: {X.shape[0]} patients, {X.shape[1]} proteins")
    return X, y, feat_cols


def build_training_data(disease_path, protein_paths, target_code, sample_n=3000):
    print("\n📂 Loading data...")
    df_patient = load_binary_csv(disease_path)
    df_protein = load_protein_data(protein_paths)

    X, y, feat_cols = build_single_disease_table(df_patient, df_protein, target_code, sample_n)

    return X, y, feat_cols