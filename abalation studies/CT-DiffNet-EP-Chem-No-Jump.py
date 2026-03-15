import os
import sys
import random
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import ConcatDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# Ensure custom modules are in the current directory
try:
    from cached_dataset import SimpleBarrierDataset
    from barrier_resnet_se2 import CNNTransformer3D
except ImportError:
    print("Warning: 'cached_dataset' or 'barrier_resnet_se2' not found. Ensure they are in the same directory.")

# ==========================================
# 1. Global Configuration and Seed Fixing
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
    print(f"Random seed set to: {seed}")

# ==========================================
# 2. Publication-Quality Plot Settings
# ==========================================
def set_publication_style(font_file='Times New Roman.ttf'):
    # 1. Strictly check if the font file exists
    if not os.path.exists(font_file):
        print(f"Warning: Font file '{font_file}' not found. Using default serif font.")
        plt.rcParams['font.family'] = 'serif'
    else:
        # 2. Load custom font
        fm.fontManager.addfont(font_file)
        prop = fm.FontProperties(fname=font_file)
        custom_font_name = prop.get_name()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [custom_font_name]
    
    # 3. Set global parameters
    plt.rcParams['mathtext.fontset'] = 'stix'       # Math font matching Times
    
    # Font sizes and line widths
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14
    
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['pdf.fonttype'] = 42 
    plt.rcParams['ps.fonttype'] = 42

# Execute style configuration
set_publication_style('Times New Roman.ttf')

# ============ EarlyStopping Class ============
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

# ============ Input Dropout Wrapper ============
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

def plot_publication_quality(y_true, y_pred, save_path='parity_plot_final.tiff'):
    """
    Generate publication-quality parity plot.
    Features:
    1. Colorbar: Keep colorbar and Low/High labels, [COMPLETELY REMOVE] minor ticks.
    2. Stats Box: [REMOVE BORDER], only two lines of text (R2 and MAE).
    3. Color Scheme: Maintain Scheme A (sophisticated cool-warm gradient).
    """
    print("Generating Final Publication-Quality Parity Plot...")
    
    # 1. Setup Canvas
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # ===== 🎨 Color Scheme (Scheme A) =====
    rgb_start = [15, 85, 105]    # Deep Teal
    rgb_end   = [230, 120, 40]   # Amber Orange
    inset_color_hex = '#0F5569'  
    inset_line_hex  = '#083D4B'
    
    color_start = np.array(rgb_start) / 255
    color_end = np.array(rgb_end) / 255
    custom_cmap = LinearSegmentedColormap.from_list('sophisticated_gradient', [color_start, color_end], N=512)
    # ==========================================
    
    # 2. Calculate Density
    xy = np.vstack([y_true, y_pred])
    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_plot, y_plot, z_plot = y_true[idx], y_pred[idx], z[idx]
        cmap = custom_cmap
        norm = LogNorm()
    except:
        x_plot, y_plot, z_plot = y_true, y_pred, 'blue'
        cmap = None
        norm = None

    # 3. Plot Scatter
    sc = ax.scatter(x_plot, y_plot, c=z_plot, cmap=cmap, norm=norm, s=45, 
                    alpha=0.9, edgecolor='none', zorder=1) 

    # 4. Reference Line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.1
    lims = [min_val - margin, max_val + margin]
    ax.plot(lims, lims, color='#333333', linestyle='--', linewidth=1.8, zorder=0)
    
    # 5. Axes Setting
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('NEB Calculated Barrier (eV)', fontsize=16, fontweight='medium', color='black') 
    ax.set_ylabel('CT-DiffNet Predicted Barrier (eV)', fontsize=16, fontweight='medium', color='black') 
    ax.tick_params(direction='in', length=5, width=1.2, labelsize=14, colors='black')

    # 6. Stats Box ([Modification]: No border, two lines)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    stats_text = (f"$R^2 = {r2:.3f}$\n" f"MAE = {mae:.3f} eV")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', 
            # edgecolor='none' removes border, keeps faint background to prevent overlap
            bbox=dict(boxstyle='square,pad=0.2', facecolor='#F8F9FA', alpha=0.8, edgecolor='none'))

    # 7. Inset Plot
    ax_inset = ax.inset_axes([0.6, 0.08, 0.38, 0.30]) 
    ax_inset.set_zorder(10)
    residuals = y_true - y_pred
    sns.histplot(residuals, ax=ax_inset, kde=True, 
                 color=inset_color_hex, stat='density', 
                 element="step", alpha=0.5, linewidth=0, 
                 line_kws={'color': inset_line_hex, 'linewidth': 1.8}) 
    ax_inset.axvline(0, color='#666666', linestyle='--', linewidth=1.2)
    ax_inset.set_ylabel('') 
    ax_inset.set_xlabel('Residuals (eV)', fontsize=11, labelpad=1, color='black') 
    ax_inset.set_yticks([]) 
    ax_inset.tick_params(axis='x', labelsize=9, direction='in', pad=2, colors='black') 
    ax_inset.patch.set_alpha(0.0) 
    for spine in ax_inset.spines.values():
        spine.set_linestyle('--')    
        spine.set_linewidth(0.8)     
        spine.set_color('#888888') 

    # 8. Colorbar ([Modification]: Completely remove tick marks)
    if cmap:
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.5) # Keep Colorbar outline
        
        cbar.set_ticks([]) 
        
        # [CRITICAL] Force turn off minor ticks (for LogNorm)
        cbar.minorticks_off()
        
        # [Double Insurance] Set all tick lengths to 0
        cbar.ax.tick_params(which='both', size=0, width=0, length=0)
        
        # Keep Low/High labels
        cbar.ax.text(0.5, -0.03, 'Low', transform=cbar.ax.transAxes, 
                     ha='center', va='top', fontsize=10, color='#444444')
        cbar.ax.text(0.5, 1.03, 'High', transform=cbar.ax.transAxes, 
                     ha='center', va='bottom', fontsize=10, color='#444444')

    plt.tight_layout()
    plt.savefig(save_path, format='tiff', dpi=300, bbox_inches='tight')
    plt.show()



# ============ User Parameters ============
# --- 1. Data Paths (Please modify to your actual paths) ---
train_files = [
    "cacheNi8Re_7.npz", 
    "cacheNi8Al_7.npz", 
    "cacheNi5Re_7.npz", 
    "cacheNi5Al_7.npz",
    "cacheNi3Re_7.npz", 
    "cacheNi3Al_7.npz", 
    "cacheNi10Re_7.npz", 
    "cacheNi10Al_7.npz",
    "cacheNi1Re_7.npz", 
    "cacheNi1Al_7.npz",
]
valtest_file = "cacheNi5Al5Re_7.npz"

# --- 2. Training Hyperparameters ---
BATCH = 32
VAL_RATIO = 0.2
SEED = 42  # Set random seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
DROPOUT_P = 0.05
WEIGHT_DECAY = 1e-4
BASE_LR = 1e-4

# --- 3. Output File Naming ---
CHECKPOINT = "best_model_5Al5Re_com_no_element.pth"  # Path to save the best model
FIG_OUT_TIFF = "parity_plot_5Al5Re_com_no_element.tiff" # Final plot output path


# ============ Main Function ============
def main():
    # 1. Fix random seed
    seed_everything(SEED)

    # 2. Dataset check and loading
    if not os.path.exists(valtest_file):
        print(f"Error: {valtest_file} not found.")
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

    # 3. Model construction
    base = CNNTransformer3D(
        in_channels=7, hidden_channels=128, num_cnn_blocks=1,
        patch_size=1, emb_dim=96, num_layers=4, num_heads=8, mlp_dim=384
    )
    model = ModelWithInputDropout(base, p=DROPOUT_P).to(DEVICE)
    
    opt = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.L1Loss() 
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=20, min_delta=1e-4, mode='min')

    # 4. Training loop
    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': [], 'lr': []}
    
    print("\nStarting training (Ablation: w/o Jump Elements)...")
    for ep in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss_acc = 0.0
        train_preds, train_trues = [], []
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # ABLATION: Masking Jump Elements (Channels 4-6)
            x[:, 4:7, :, :, :] = 0.0
            
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
        
        # --- Val ---
        model.eval()
        val_loss_acc = 0.0
        val_preds, val_trues = [], []
        
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                # ABLATION: Masking Jump Elements (Channels 4-6)
                x[:, 4:7, :, :, :] = 0.0
                
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

        # --- Save Best Model ---
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

    # ============ Testing & Visualization ============
    print(f"\nTraining finished. Loading best model from {CHECKPOINT} for testing...")
    
    # 5. Load best model for testing
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.base.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    te_preds, te_trues = [], []
    with torch.no_grad():
        for x, y in te_loader:
            x = x.to(DEVICE)
            # ABLATION: Masking Jump Elements
            x[:, 4:7, :, :, :] = 0.0
            
            out = model(x)
            p = out[0].squeeze(-1).cpu() if isinstance(out, tuple) else out.squeeze(-1).cpu()
            te_preds.append(p)
            te_trues.append(y)
            
    y_pred = torch.cat(te_preds).numpy()
    y_true = torch.cat(te_trues).numpy()
    
    # Metrics (Overall)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = r2_score(y_true, y_pred)
    
    print(f"Final Test Results (Overall): MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # ====================================================
    THRESHOLD = 1.5
    high_mask = y_true > THRESHOLD
    
    if np.any(high_mask):
        y_true_high = y_true[high_mask]
        y_pred_high = y_pred[high_mask]
        
        mae_high = mean_absolute_error(y_true_high, y_pred_high)
        rmse_high = np.sqrt(mean_squared_error(y_true_high, y_pred_high))
        
        print(f"\ High-Barrier (> {THRESHOLD} eV) Results:")
        print(f"   Sample Count: {np.sum(high_mask)}")
        print(f"   High-Barrier MAE  = {mae_high:.4f} eV")
        print(f"   High-Barrier RMSE = {rmse_high:.4f} eV\n")
    else:
        print(f"\nWarning: No samples found > {THRESHOLD} eV in the test set.\n")
    # ====================================================
    
    # 6. Call gradient background plotting function
    plot_publication_quality(y_true, y_pred, save_path=FIG_OUT_TIFF)

if __name__ == "__main__":
    main()