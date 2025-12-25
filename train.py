# train.py
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from datetime import datetime
import time
from model import TabularHRM, TabularHRMConfig
from data_processing import load_datasets

def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    total_batches = len(loader)
    milestones = {1, 25, 50, 95}
    printed = set()  # to avoid duplicate prints

    for i, batch in enumerate(loader, start=1):
        # Calculate progress percentage
        progress = int((i / total_batches) * 100)

        # Print only selected milestones
        if progress in milestones and progress not in printed:
            print(f"Epoch {epoch:03d} [Train] {progress}%")
            printed.add(progress)

        # Move tensors to device
        batch["num"] = batch["num"].to(device)
        batch["bin"] = batch["bin"].to(device)
        batch["cat"] = [c.to(device) for c in batch["cat"]]
        batch["y"] = batch["y"].to(device)

        # Forward + backward + optimize
        optimizer.zero_grad()
        logits = model(batch)["y"].squeeze(1)
        loss = criterion(logits, batch["y"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch["y"])

    # Return average loss for the epoch
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device, epoch):
    model.eval()
    ys, preds = [], []

    for batch in loader:
        # Move tensors to device, handling the list of categorical tensors
        batch["num"] = batch["num"].to(device)
        batch["bin"] = batch["bin"].to(device)
        batch["cat"] = [c.to(device) for c in batch["cat"]]
        batch["y"] = batch["y"].to(device)

        logits = model(batch)["y"].squeeze(1)
        probs = torch.sigmoid(logits)

        ys.append(batch["y"].cpu())
        preds.append(probs.cpu())

    y = torch.cat(ys).numpy()
    p = torch.cat(preds).numpy()
    return roc_auc_score(y, p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4)

    args = parser.parse_args()

    # Record start time
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"Training started at: {start_datetime}")
    print(f"{'='*60}\n")

    print("Loading datasets...")
    train_ds, val_ds, meta = load_datasets(args.data_dir)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Features - Numeric: {meta['num_dim']}, Binary: {meta['bin_dim']}, Categorical: {len(meta['cat_cols'])}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    cfg = TabularHRMConfig(
        numeric_dim=meta["num_dim"],
        binary_dim=meta["bin_dim"],
        cat_vocab_sizes=meta["cat_sizes"],
        cat_emb_dims=meta["cat_emb_dims"],
        hidden_size=128,
        output_heads={"y": 1},
    )

    print(f"\nInitializing model on {args.device}...")
    model = TabularHRM(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = 0.0
    epochs_no_improve = 0

    patience = args.early_stop_patience
    min_delta = args.early_stop_delta

    os.makedirs("checkpoints", exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.device, epoch)
        val_auc = eval_epoch(model, val_loader, args.device, epoch)


        if val_auc > best_auc + min_delta:
            best_auc = val_auc
            epochs_no_improve = 0

            torch.save(model.state_dict(), "checkpoints/best.pt")
            torch.save(meta, "checkpoints/meta.pt")

            improvement = "*"
        else:
            epochs_no_improve += 1
            improvement = ""

        print(f"Epoch {epoch:03d} | Loss {train_loss:.4f} | AUC {val_auc:.4f} {improvement}")

        # ---- EARLY STOPPING CHECK ----
        if epochs_no_improve >= patience:
            print(
                f"\nEarly stopping triggered after {epoch} epochs "
                f"(no AUC improvement for {patience} consecutive epochs)."
            )
            break
        
    # Record end time
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time
    
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print("=" * 60)
    print(f"Training complete! Best validation AUC: {best_auc:.4f}")
    print(f"\nTraining ended at: {end_datetime}")
    if hours > 0:
        print(f"Total time taken: {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"Total time taken: {minutes}m {seconds}s")
    else:
        print(f"Total time taken: {seconds}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()