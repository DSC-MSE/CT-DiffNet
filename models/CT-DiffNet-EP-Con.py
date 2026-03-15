#!/usr/bin/env python3
"""
CT-DiffNet Specific Evaluation Script
======================================================
Task: Train the model on a specific subsystem (e.g., 1Al1Re) 
      and evaluate on a higher-concentration target (e.g., 5Al5Re).
"""

import os
import sys
import random
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import ConcatDataset, DataLoader, random_split

# Ensure custom modules are in the current directory
try:
    from cached_dataset import SimpleBarrierDataset
    from barrier_resnet_se2 import CNNTransformer3D
except ImportError:
    print("Warning: 'cached_dataset' or 'barrier_resnet_se2' not found. Ensure they are in the same directory.")
    sys.exit(1)

# ==========================================
# 1. Global Configuration
# ==========================================
def seed_everything(seed=42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Random seed fixed: {seed}")

# ==========================================
# 2. Utility Classes
# ==========================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0
        self.best_epoch = 0

    def __call__(self, current, epoch=None):
        if self.best is None:
            self.best = current
            self.best_epoch = epoch if epoch else 0
            return False

        if self.mode == 'min':
            improved = (current < self.best - self.min_delta)
        else:
            improved = (current > self.best + self.min_delta)

        if improved:
            self.best = current
            self.num_bad_epochs = 0
            self.best_epoch = epoch if epoch else 0
            return False
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                return True
            return False

class ModelWithInputDropout(nn.Module):
    def __init__(self, base_model, p):
        super().__init__()
        self.base = base_model
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.drop(x)
        return self.base(x)
    
    @property
    def return_attention(self):
        return self.base.return_attention
    
    @return_attention.setter
    def return_attention(self, value):
        self.base.return_attention = value

# ==========================================
# 3. User Parameters
# ==========================================
train_files = [
    "cacheNi1Al1Re_7.npz", 
]
valtest_file = "cacheNi5Al5Re_7.npz"

BATCH = 32
VAL_RATIO = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
DROPOUT_P = 0.05
WEIGHT_DECAY = 1e-4
BASE_LR = 1e-4

CHECKPOINT = "best_model_con_5Al5Re2.pth"

# ==========================================
# 4. Main Training Loop
# ==========================================
def main():
    seed_everything(SEED)

    # --- 4.1 Dataset Setup ---
    if not os.path.exists(valtest_file):
        print(f"Error: Validation/Test file {valtest_file} not found.")
        return

    print("Loading datasets...")
    train_ds_list = [SimpleBarrierDataset(p) for p in train_files if os.path.exists(p)]
    if not train_ds_list:
        print("Error: No training files found.")
        return
        
    train_ds = ConcatDataset(train_ds_list)
    valtest_ds = SimpleBarrierDataset(valtest_file)
    n_total = len(valtest_ds)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_val

    g = torch.Generator().manual_seed(SEED)
    val_ds, test_ds = random_split(valtest_ds, [n_val, n_test], generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    te_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Device: {DEVICE}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    # --- 4.2 Model Construction ---
    base = CNNTransformer3D(
        in_channels=7, hidden_channels=128, num_cnn_blocks=1,
        patch_size=1, emb_dim=96, num_layers=4, num_heads=8, mlp_dim=384
    )
    model = ModelWithInputDropout(base, p=DROPOUT_P).to(DEVICE)
    
    opt = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.L1Loss() 
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=20, min_delta=1e-4, mode='min')

    # --- 4.3 Training Process ---
    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': [], 'lr': []}
    
    print("\nStarting training...")
    for ep in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        train_loss_acc = 0.0
        train_preds, train_trues = [], []
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            pred = out[0].squeeze(-1) if isinstance(out, tuple) else out.squeeze(-1)
            
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            
            train_loss_acc += loss.item() * x.size(0)
            train_preds.append(pred.detach().cpu())
            train_trues.append(y.detach().cpu())
            
        train_loss = train_loss_acc / len(train_ds)
        train_r2 = r2_score(torch.cat(train_trues).numpy(), torch.cat(train_preds).numpy())
        
        # Validation Phase
        model.eval()
        val_loss_acc = 0.0
        val_preds, val_trues = [], []
        
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                pred = out[0].squeeze(-1) if isinstance(out, tuple) else out.squeeze(-1)
                val_loss_acc += loss_fn(pred, y).item() * x.size(0)
                val_preds.append(pred.cpu())
                val_trues.append(y.cpu())
                
        val_loss = val_loss_acc / len(val_ds)
        val_r2 = r2_score(torch.cat(val_trues).numpy(), torch.cat(val_preds).numpy())
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        print(f"Ep {ep:03d} | Train MAE: {train_loss:.4f}, R2: {train_r2:.4f} | Val MAE: {val_loss:.4f}, R2: {val_r2:.4f}")

        # Save Best Model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.base.state_dict(),
                'best_val': best_val,
                'optimizer_state_dict': opt.state_dict(),
                'history': history
            }, CHECKPOINT)

        if early_stopping(val_loss, epoch=ep):
            print(f"Early stopping triggered at epoch {ep}")
            break

    # ==========================================
    # 5. Testing Phase
    # ==========================================
    print(f"\nTraining finished. Loading best model from {CHECKPOINT} for testing...")
    
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.base.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    te_preds, te_trues = [], []
    with torch.no_grad():
        for x, y in te_loader:
            x = x.to(DEVICE)
            out = model(x)
            p = out[0].squeeze(-1).cpu() if isinstance(out, tuple) else out.squeeze(-1).cpu()
            te_preds.append(p)
            te_trues.append(y)
            
    y_pred = torch.cat(te_preds).numpy()
    y_true = torch.cat(te_trues).numpy()
    
    # Metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = r2_score(y_true, y_pred)
    
    print(f"Final Test Results: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    main()