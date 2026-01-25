import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

def _to_python_scalar(x):
    # unwrap numpy scalars/0-d arrays and 1-elem arrays
    if isinstance(x, np.ndarray):
        if x.shape == ():          # 0-d array
            x = x.item()
        elif x.size == 1:          # [1] array
            x = x.reshape(()).item()
    # unwrap torch scalars too (just in case)
    if torch.is_tensor(x) and x.numel() == 1:
        x = x.item()
    return x

# =====================================================
# MODEL
# =====================================================

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequences.
    Expects x: [batch, seq_len, d_model]
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.cos(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class SpiralTransformer(nn.Module):
    """
    Encoder-only transformer for spiral embeddings.

    Input:
        x: [batch, seq_len, embed_dim]
        padding_mask: [batch, seq_len] bool (True = PAD)

    Outputs (dict):
        
    """

    def __init__(
        self,
        embed_dim: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()

        self.input_proj = nn.Linear(embed_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # --- global k for whole spiral ---
        self.k_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        
        # --- global k tight for whole spiral ---
        self.k_tight_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),   # sin, cos
        )
        
        self.angle_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, padding_mask=None, theta_center=None, theta_halfspan=None):
        # x: [B, P, embed_dim]
        x = self.input_proj(x)      # [B, P, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        enc = self.encoder(x, src_key_padding_mask=padding_mask)  # [B, P, d_model]
    
        # mask-aware mean pool for global k
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1)        # [B,P,1]
            enc_masked = enc * valid
            lengths = valid.sum(dim=1).clamp(min=1)      # [B,1]
            pooled = enc_masked.sum(dim=1) / lengths     # [B,d_model]
        else:
            pooled = enc.mean(dim=1)

        k_raw = self.k_head(pooled).squeeze(-1)          # [B]
        k_pred = k_raw  # can be negative
        # k_pred = F.softplus(k_raw) # if you need k >= 0:
        
        k_tight_raw = self.k_tight_head(pooled).squeeze(-1)  # [B]
        k_tight_pred = k_tight_raw  # can be negative        
        
        scores = self.angle_pool(enc).squeeze(-1)  # [B,P]

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -1e9)
            
        attn = torch.softmax(scores, dim=1)                 # [B,P]
        pooled_angle = (enc * attn.unsqueeze(-1)).sum(dim=1)  # [B,d_model]

        ang_raw = self.angle_head(pooled_angle)             # [B,2]
        # ang_vec = F.normalize(ang_raw, dim=-1)     # unit vector
        # sin_pred = ang_vec[:, 0]
        # cos_pred = ang_vec[:, 1]

        return {
            "k_pred": k_pred,
            "k_tight_pred": k_tight_pred,
            # "theta_tight": (sin_pred, cos_pred),
            "ang_raw": ang_raw,
        }

# =====================================================
# DATASET
# =====================================================

class SpiralDataset(Dataset):

    def __init__(
        self,
        file_paths: List[Path] = [],
        max_num_per_class: int = 10000,
        validation_size: int = 0,
    ):
        super().__init__()
        self.items = []
        if validation_size > 0:
            assert max_num_per_class + validation_size <= len(file_paths[0]), \
                "Not enough data for the requested training + validation size."
            for class_paths in file_paths:
                selected_paths = class_paths[max_num_per_class:max_num_per_class+validation_size]
                self.items.extend(selected_paths)
        else:           
            for class_paths in file_paths:
                selected_paths = class_paths[:max_num_per_class]
                if isinstance(selected_paths, str):
                    selected_paths = [selected_paths]

                self.items.extend(selected_paths)

        if not self.items:
            raise RuntimeError("No data passed.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.items[idx]
        d = np.load(file_path, allow_pickle=True)
        theta = d["theta"]                      # [T]
        r = d["r"]                              # [T]
        k = d["k"]                              # scalar
        k_tight = d["k_tight"]                  # scalar
        is_tight = torch.tensor(0, dtype=torch.float32)

        theta_tight = 0.0
        if "theta_tight" in d:
            raw = _to_python_scalar(d["theta_tight"])
            if raw is not None:
                theta_tight = float(raw)
                is_tight = torch.tensor(1, dtype=torch.float32)
        
                
        seq_emb = d["embedding"]                # [P, d_emb]; P = num_patches
        T = len(theta)

        L, d_emb = seq_emb.shape
        
        edges = np.linspace(0, T, L + 1).astype(int)

        seq_emb = torch.from_numpy(seq_emb).float()                         # [L, d_emb]

        return {
            "seq_emb": seq_emb,
            "k": k,
            "k_tight": k_tight,
            "theta_tight": theta_tight,
            "is_tight": is_tight,
            "name": file_path.stem,
            # time-step level info for plotting:
            "theta": torch.from_numpy(theta).float(),                   # [T]
            "r": torch.from_numpy(r).float(),                           # [T]
        }

def spiral_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads variable-length patch-level sequences in a batch of spirals.

    Expects each item to have:
      - "seq_emb":       [L, d_emb]
      - "flat_onehot":   [L]
      - "tight_onehot":  [L]
      - "tightness":     [L]
      - "normality":     [L]
      - "k":             scalar
      - "name":          str
      - "theta":         [T]
      - "r":             [T]
      - "flat_onehot_ts":   [T]
      - "tight_onehot_ts":  [T]
      - "tightness_ts":     [T]
      - "normality_ts":     [T]
    """
    batch_size = len(batch)
    lengths = [item["seq_emb"].shape[0] for item in batch]
    max_len = max(lengths)
    d_emb = batch[0]["seq_emb"].shape[1]

    # Patch-level tensors
    seq_emb_batch = torch.zeros(batch_size, max_len, d_emb, dtype=torch.float32)

    # Mask: True = PAD
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    # k is scalar per spiral
    k_batch = torch.tensor([float(item["k"]) for item in batch], dtype=torch.float32)
    k_tight_batch = torch.tensor([float(item["k_tight"]) for item in batch], dtype=torch.float32)
    theta_tight_batch = torch.tensor([float(item["theta_tight"]) for item in batch], dtype=torch.float32)
    is_tight_batch = torch.tensor([int(item["is_tight"]) for item in batch], dtype=torch.float32)

    # Non-padded / meta stuff
    names = [item["name"] for item in batch]
    thetas = [item["theta"] for item in batch]                 # list[tensor] (T_i)
    rs = [item["r"] for item in batch]

    # Fill padded tensors
    for i, item in enumerate(batch):
        L = item["seq_emb"].shape[0]

        seq_emb_batch[i, :L, :] = item["seq_emb"]

        padding_mask[i, :L] = False  # these positions are real, not PAD

    return {
        "seq_emb": seq_emb_batch,             # [B, L_max, d_emb]
        "k": k_batch,                           # [B]
        "k_tight": k_tight_batch,                     # [B]
        "theta_tight": theta_tight_batch,             # [B]
        "is_tight": is_tight_batch,                 # [B]
        "padding_mask": padding_mask,           # [B, L_max]
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "names": names,
        "theta": thetas,                      # list[tensor] (T_i)
        "r": rs,
    }

# =====================================================
# TRAINING
# =====================================================

@torch.no_grad()
def _run_eval(model, loader, device, k_loss_fn=None,
              w_k=None, w_k_tight=None, w_angle=None):
    model.eval()

    total = 0.0

    for batch in loader:
        seq_emb = batch["seq_emb"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        
        k_true = batch["k"].to(device).float()
        k_tight_true = batch["k_tight"].to(device).float()
        theta_tight_true = batch["theta_tight"].to(device).float()
        is_tight = batch["is_tight"].to(device).float()
        
        if k_true.ndim == 2 and k_true.shape[-1] == 1:
            k_true = k_true.squeeze(-1)
        if k_tight_true.ndim == 2 and k_tight_true.shape[-1] == 1:
            k_tight_true = k_tight_true.squeeze(-1)

        out = model(seq_emb, padding_mask=padding_mask)

        k_pred              = out["k_pred"]
        k_tight_pred        = out["k_tight_pred"]
        ang_raw             = out["ang_raw"].view(-1)
        
        loss_k          = k_loss_fn(k_pred, k_true)
        loss_k_tight    = k_loss_fn(k_tight_pred, k_tight_true)
        
        mask = (is_tight > 0.5).float().view(-1)
        theta = theta_tight_true.float().view(-1)

        per = (ang_raw - theta) ** 2
        loss_angle = (per * mask).sum() / mask.sum().clamp(min=1.0)

        loss = (
            w_k         * loss_k       +
            w_k_tight   * loss_k_tight +
            w_angle     * loss_angle
        )

        total += float(loss.item())

    return total / max(1, len(loader))

def train_one_epoch(model, loader, optimizer, device, val_loader=None,
                    k_loss_fn=nn.SmoothL1Loss(), w_k=0.5, w_k_tight=0.5, w_angle=0.7):
    model.train()
    total_loss = 0.0

    for batch in loader:
        seq_emb = batch["seq_emb"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        
        k_true = batch["k"].to(device).float()
        k_tight_true = batch["k_tight"].to(device).float()
        theta_tight_true = batch["theta_tight"].to(device).float()
        is_tight = batch["is_tight"].to(device).float()
        
        if k_true.ndim == 2 and k_true.shape[-1] == 1:
            k_true = k_true.squeeze(-1)
        if k_tight_true.ndim == 2 and k_tight_true.shape[-1] == 1:
            k_tight_true = k_tight_true.squeeze(-1)
    
        optimizer.zero_grad(set_to_none=True)

        out = model(seq_emb, padding_mask=padding_mask)

        k_pred              = out["k_pred"]
        k_tight_pred        = out["k_tight_pred"]
        ang_raw             = out["ang_raw"].view(-1)     # [B]
        
        loss_k          = k_loss_fn(k_pred, k_true)
        loss_k_tight    = k_loss_fn(k_tight_pred, k_tight_true)
        
        mask = (is_tight > 0.5).float().view(-1)          # [B]
        theta = theta_tight_true.float().view(-1)         # [B]

        per = (ang_raw - theta) ** 2                      # [B]
        loss_angle = (per * mask).sum() / mask.sum().clamp(min=1.0)
        
        loss = (
            w_k         * loss_k       +
            w_k_tight   * loss_k_tight +
            w_angle     * loss_angle
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    train_loss = total_loss / max(1, len(loader))

    val_loss = None
    if val_loader is not None:
        val_loss = _run_eval(
            model, val_loader, device, k_loss_fn=k_loss_fn,
            w_k=w_k, w_k_tight=w_k_tight, w_angle=w_angle
        )

    return train_loss, val_loss

# =====================================================
# LOGGING
# =====================================================
def log_model_metadata(
    model_name: str,
    dataset_size: int,
    num_epochs: int,
    time_taken,
    final_train_loss: float,
    final_val_loss: float = 0.0,
    log_path: Path = Path("../models/MODEL_DESCRIPTIONS.md"),
) -> None:
    """
    Append a new model entry to the markdown log file.
    Follows the template:

    <ul>
    <li>
    <i><b>'model_name'</b></i> - model created on 'date', on a dataset of size 'size'.<br>Final training loss: 'training_loss'<br>Final validation loss: 'validation_loss'
    </li>
    </ul>
    """
    log_path.parent.mkdir(exist_ok=True, parents=True)

    today_str = datetime.now().strftime("%Y-%m-%d")
    entry = (
        "<ul>\n"
        "<li>\n"
        f"<i><b>'{model_name}'</b></i> - model created on '{today_str}' by transformer5, "
        f"on a dataset of size '{dataset_size}', with {num_epochs} epochs.<br>"
        f"Time taken: '{time_taken}'<br>"
        f"Final training loss: '{final_train_loss:.4f}'<br>"
        f"Final validation loss: '{final_val_loss:.4f}'\n"
        "</li>\n"
        "</ul>\n\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

# =====================================================
# MAIN
# =====================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    MAX_NUM_PER_CLASS = 65000
    VALIDATION_SIZE = 6500

    data_dirs = [Path("../data/normal"), Path("../data/tight"), Path("../data/spiky"), Path("../data/flat")]
    
    spiral_paths = []
    for d in data_dirs:
        files = sorted(d.glob("*.npz"))
        spiral_paths.append(files)
    
    train_dataset = SpiralDataset(
        file_paths=spiral_paths,
        max_num_per_class=MAX_NUM_PER_CLASS,
    )
    
    val_dataset = SpiralDataset(
        file_paths=spiral_paths,
        max_num_per_class=MAX_NUM_PER_CLASS,
        validation_size=VALIDATION_SIZE,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=spiral_collate,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=spiral_collate,
    )

    # Infer embed_dim from one sample
    sample = train_dataset[0]
    d_emb = sample["seq_emb"].shape[1]

    model = SpiralTransformer(
        embed_dim=d_emb,
        d_model=256,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        dropout=0.3,
        max_seq_len=1024,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # ---- Training loop ----
    NUM_EPOCHS = 70
    
    # Early stopping
    best_val_loss = float("inf")
    steps_wo_improve = 0
    STEPS_LIM = 10
    
    start_time = datetime.now()
    print("Starting training...", flush=True)
    train_losses, val_losses = [], []
    for epoch in range(NUM_EPOCHS):
        train_loss, val_loss = train_one_epoch(model, train_loader, optimizer, device, val_loader=val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: loss = {train_loss:.4f}, validation loss = {val_loss:.4f}", flush=True)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            steps_wo_improve = 0
        else:
            steps_wo_improve += 1
            if steps_wo_improve >= STEPS_LIM:
                print(f"No improvement for {STEPS_LIM} epochs. Stopping early.", flush=True)
                break
        
    save_path = Path("../models")
    save_path.mkdir(exist_ok=True, parents=True)
    model_name = f"sp_trans_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt"
    model_file = save_path / model_name
    torch.save(model.state_dict(), model_file)
    
    end_time = datetime.now()
    time_taken = end_time - start_time
    total_seconds = int(time_taken.total_seconds())

    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60

    print(f"Training completed in {h:02d}:{m:02d}:{s:02d}.")

    # --- Log model metadata ---
    log_model_metadata(
        model_name=model_name,
        dataset_size=len(train_dataset),
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
        num_epochs=NUM_EPOCHS,
        time_taken=time_taken,
    )
    
    # --- Save train loss plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o', color='navy', label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, marker='o', color="orange", label='Validation Loss')
    plt.legend()
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.grid(True)
    plt.savefig(save_path / f"training_loss_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png")
    
if __name__ == "__main__":
    main()