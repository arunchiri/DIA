# data_processing.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class TabularDataset(Dataset):
    def __init__(self, df, num_cols, bin_cols, cat_cols, scaler=None, train=True):
        self.num = df[num_cols].values.astype("float32") if num_cols else None
        self.bin = df[bin_cols].values.astype("float32") if bin_cols else None
        self.cats = [df[c].values.astype("int64") for c in cat_cols] if cat_cols else []
        self.y = None

        if train:
            self.y = torch.tensor(df["diagnosed_diabetes"].values, dtype=torch.float32)

        if scaler and self.num is not None:
            self.num = scaler.transform(self.num)

        self.num = torch.tensor(self.num) if self.num is not None else torch.zeros(len(df), 0)
        self.bin = torch.tensor(self.bin) if self.bin is not None else torch.zeros(len(df), 0)
        self.cats = [torch.tensor(c) for c in self.cats]

    def __len__(self):
        return len(self.num)

    def __getitem__(self, idx):
        out = {
            "num": self.num[idx],
            "bin": self.bin[idx],
            "cat": [c[idx] for c in self.cats],
        }
        if self.y is not None:
            out["y"] = self.y[idx]
        return out

def load_datasets(data_dir):
    df = pd.read_csv(f"{data_dir}/train.csv")

    # Identify column types
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in ["id", "diagnosed_diabetes"]]
    bin_cols = [c for c in df.columns if c not in ["id", "diagnosed_diabetes"] and set(df[c].dropna().unique()) <= {0, 1}]
    num_cols = [c for c in df.columns if c not in cat_cols + bin_cols + ["id", "diagnosed_diabetes"]]

    # Encode categorical columns
    for c in cat_cols:
        df[c] = df[c].astype("category").cat.codes

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["diagnosed_diabetes"], random_state=42)

    # Fit scaler on training data
    scaler = StandardScaler().fit(train_df[num_cols].values) if num_cols else None

    # Create datasets
    train_ds = TabularDataset(train_df, num_cols, bin_cols, cat_cols, scaler, train=True)
    val_ds = TabularDataset(val_df, num_cols, bin_cols, cat_cols, scaler, train=True)

    # Prepare metadata
    meta = {
        "num_dim": len(num_cols),
        "bin_dim": len(bin_cols),
        "cat_sizes": [df[c].nunique() + 1 for c in cat_cols],
        "cat_emb_dims": [min(32, (n + 1) // 2) for n in [df[c].nunique() for c in cat_cols]],
        "scaler": scaler,
        "num_cols": num_cols,
        "bin_cols": bin_cols,
        "cat_cols": cat_cols,
    }

    return train_ds, val_ds, meta