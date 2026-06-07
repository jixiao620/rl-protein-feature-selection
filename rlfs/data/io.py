# rlfs/data/io.py
from __future__ import annotations

import pandas as pd


def load_binary_csv(path: str) -> pd.DataFrame:
    print(f"Loading disease binary table: {path}")
    return pd.read_csv(path, compression="gzip")


def load_protein_data(paths: list[str]) -> pd.DataFrame:
    print("Loading Olink protein data blocks...")
    dfs = []
    header_cols = None

    for i, p in enumerate(paths):
        print(f"Reading {p} ...")
        if i == 0:
            df_part = pd.read_csv(p, compression="gzip")
            if "userID" not in df_part.columns:
                df_part.rename(columns={df_part.columns[0]: "userID"}, inplace=True)
            header_cols = df_part.columns
            print(f"Header detected: {len(header_cols)} columns")
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

    bad_cols = [
        c for c in df_all.columns
        if c.startswith("Unnamed") or c.replace(".", "", 1).isdigit()
    ]
    if bad_cols:
        print(f"Removing {len(bad_cols)} invalid columns")
        df_all = df_all.drop(columns=bad_cols)

    print(f"Loaded protein data: {df_all.shape[0]} patients × {df_all.shape[1]} features")
    return df_all
