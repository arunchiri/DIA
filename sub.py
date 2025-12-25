# sub.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
import time
from model import TabularHRM, TabularHRMConfig
from data_processing import TabularDataset

@torch.no_grad()
def main():
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"Inference started at: {start_datetime}")
    print(f"{'='*60}\n")

    # Load test data and metadata
    print("Loading test data...")
    df = pd.read_csv("dataset/test.csv")
    print(f"Test samples: {len(df)}")
    
    print("Loading model metadata...")
    meta = torch.load("checkpoints/meta.pt")

    # Encode categorical columns (ensure they match training encoding)
    print("Processing categorical features...")
    for c in meta["cat_cols"]:
        if c in df.columns:
            df[c] = df[c].astype("category").cat.codes
        else:
            print(f"Warning: Column {c} not found in test data")

    # Create dataset
    ds = TabularDataset(
        df,
        meta["num_cols"],
        meta["bin_cols"],
        meta["cat_cols"],
        meta["scaler"],
        train=False,
    )
    loader = DataLoader(ds, batch_size=512)

    # Initialize model
    print("Loading trained model...")
    cfg = TabularHRMConfig(
        numeric_dim=meta["num_dim"],
        binary_dim=meta["bin_dim"],
        cat_vocab_sizes=meta["cat_sizes"],
        cat_emb_dims=meta["cat_emb_dims"],
        hidden_size=128,
        output_heads={"y": 1},
    )

    model = TabularHRM(cfg)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    model.eval()

    # Generate predictions
    print("Generating predictions...")
    preds = []
    for batch in loader:
        logits = model(batch)["y"].squeeze(1)
        preds.append(torch.sigmoid(logits))

    preds = torch.cat(preds).numpy()

    # Create submission file
    print("Creating submission file...")
    sub = pd.DataFrame({
        "id": df["id"],
        "diagnosed_diabetes": preds
    })
    sub.to_csv("submission.csv", index=False)
    
    # Record end time
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time
    
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"Submission file created with {len(sub)} predictions")
    print(f"\nInference ended at: {end_datetime}")
    if minutes > 0:
        print(f"Total time taken: {minutes}m {seconds}s")
    else:
        print(f"Total time taken: {seconds}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()