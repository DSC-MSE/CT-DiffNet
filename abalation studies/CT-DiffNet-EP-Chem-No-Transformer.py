#!/usr/bin/env python3
"""
======================================================
Task: Compare CT-DiffNet (Hybrid) vs. Pure CNN (Baseline).
Key Metric: Absolute Prediction Error (MAE) on High vs Low barriers.
Style: High-impact publication standard (Times New Roman, TIFF).
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
from torch.utils.data import DataLoader

# ==========================================
# 1. Import Models and Datasets
# ==========================================
try:
    from cached_dataset import SimpleBarrierDataset
    from barrier_resnet_se2 import CNNTransformer3D
    from CNN3D_NoTransformer import CNN3D_NoTransformer
except ImportError as e:
    print(f"Error: Missing required modules ({e}). Please ensure all model files are in the current directory.")
    exit(1)

# ==========================================
# 2. Style and Configuration
# ==========================================
def set_publication_style(font_file='Times New Roman.ttf'):
    if os.path.exists(font_file):
        fm.fontManager.addfont(font_file)
        prop = fm.FontProperties(fname=font_file)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [prop.get_name()]
    else:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
    
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['figure.autolayout'] = True
    sns.set_style("ticks")

set_publication_style('Times New Roman.ttf')

# Paths & Settings
TEST_FILE = "cacheNi5Al5Re_7.npz"
HYBRID_PTH = "best_model_5Al5Re_composition.pth"
CNN_PTH = "cnn_only_baseline.pth"
SAVE_NAME = "fig_model_comparison_refined.tiff"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. Inference Function
# ==========================================
def get_predictions(model, loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            if isinstance(out, tuple): 
                out = out[0]
            preds.append(out.squeeze().cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)

# ==========================================
# 4. Main Logic: Evaluation and Comparison
# ==========================================
def run_comparison():
    print(f"Loading Test Data: {TEST_FILE}")
    if not os.path.exists(TEST_FILE):
        print("Error: Test file not found.")
        return None
        
    dataset = SimpleBarrierDataset(TEST_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- A. Hybrid Model (Ours) ---
    print("Evaluating Hybrid Model (CT-DiffNet)...")
    model_hybrid = CNNTransformer3D(
        in_channels=7, hidden_channels=128, num_cnn_blocks=1, 
        patch_size=1, num_layers=4, num_heads=8, mlp_dim=384, emb_dim=96 
    ).to(DEVICE)
    
    current_emb_dim = 96
    target_num_patches = 64
    model_hybrid.pos_embed = torch.nn.Parameter(
        torch.zeros(1, target_num_patches, current_emb_dim).to(DEVICE)
    )
    
    if os.path.exists(HYBRID_PTH):
        ckpt = torch.load(HYBRID_PTH, map_location=DEVICE)
        model_hybrid.load_state_dict(ckpt['model_state_dict'], strict=False)
        y_pred_hyb, y_true = get_predictions(model_hybrid, loader)
    else:
        print(f"Warning: {HYBRID_PTH} not found! Skipping.")
        return None

    # --- B. Pure CNN Model (Baseline) ---
    print("Evaluating Pure CNN Model (Baseline)...")
    model_cnn = CNN3D_NoTransformer(
        in_channels=7, 
        hidden_channels=128, 
        num_cnn_blocks=1,
        emb_dim=96 
    ).to(DEVICE)
    
    if os.path.exists(CNN_PTH):
        ckpt = torch.load(CNN_PTH, map_location=DEVICE)
        model_cnn.load_state_dict(ckpt['model_state_dict'], strict=False)
        y_pred_cnn, _ = get_predictions(model_cnn, loader)
    else:
        print(f"Warning: {CNN_PTH} not found! Comparison will be invalid.")
        return None

    # --- C. Build DataFrame ---
    df = pd.DataFrame({
        'True': y_true,
        'AE_Hybrid': np.abs(y_true - y_pred_hyb),
        'AE_CNN': np.abs(y_true - y_pred_cnn)
    })
    
    # Define barrier regions (Threshold = 1.5 eV)
    df['Region'] = df['True'].apply(
        lambda x: 'High Barrier (>1.5 eV)' if x >= 1.5 else 'Low Barrier (<1.5 eV)'
    )
    
    return df

# ==========================================
# 5. Plotting Function
# ==========================================
def plot_comparison_refined(df):
    if df is None: return

    # Data Preparation
    df_melt = pd.melt(df, 
                      id_vars=['Region'], 
                      value_vars=['AE_CNN', 'AE_Hybrid'],
                      var_name='Model', 
                      value_name='Absolute Error')
    
    df_melt['Model'] = df_melt['Model'].replace({
        'AE_CNN': 'CNN-only model', 
        'AE_Hybrid': 'CT-DiffNet model'
    })

    # Canvas Setup
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Color Palettes
    my_palette = {"CNN-only model": "#95A5A6", "CT-DiffNet model": "#E74C3C"}
    mean_palette = {"CNN-only model": "#2C3E50", "CT-DiffNet model": "#7B241C"}

    # Draw Boxplot and Stripplot
    sns.stripplot(x="Region", y="Absolute Error", hue="Model",
                  data=df_melt, palette=my_palette,
                  dodge=True, alpha=0.15, size=4, jitter=0.2, ax=ax, zorder=0)
                  
    sns.boxplot(x="Region", y="Absolute Error", hue="Model", 
                data=df_melt, palette=my_palette, 
                showfliers=False, 
                width=0.5, 
                linewidth=1.5, 
                ax=ax)

    # Draw Mean Diamonds
    sns.pointplot(x="Region", y="Absolute Error", hue="Model", 
                  data=df_melt, 
                  dodge=0.25, 
                  join=False, 
                  markers=["D", "D"], 
                  scale=0.9, 
                  palette=mean_palette,
                  err_kws={'linewidth': 0},
                  ax=ax)

    # Formatting & Styling
    font_axis = {'family': 'Times New Roman', 'weight': 'bold', 'size': 22}
    ax.set_xlabel('Migration Barrier Region', fontdict=font_axis, labelpad=10)
    ax.set_ylabel('Absolute Prediction Error (eV)', fontdict=font_axis, labelpad=10)

    for label in ax.get_xticklabels():
        label.set_fontproperties(fm.FontProperties(fname='Times New Roman.ttf', size=20))
    
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    for label in ax.get_yticklabels():
        label.set_fontproperties(fm.FontProperties(fname='Times New Roman.ttf', size=20))

    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    
    # Legend Settings
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[2:4], labels[2:4], 
                    loc='upper right', 
                    frameon=False)
    
    for text in leg.get_texts():
        text.set_fontfamily('Times New Roman')
        text.set_fontsize(18)
    leg.get_frame().set_alpha(0.8)

    # Performance Improvement Annotation
    high_barrier_data = df[df['True'] >= 1.5]
    if not high_barrier_data.empty:
        mean_cnn = high_barrier_data['AE_CNN'].mean()
        mean_hyb = high_barrier_data['AE_Hybrid'].mean()
        improvement = (mean_cnn - mean_hyb) / mean_cnn * 100
        
        txt = ax.text(1.15, mean_hyb*0.95, f"↓{improvement:.1f}% ", 
                ha='center', va='top', 
                fontdict={'family': 'Times New Roman', 'color': 'white', 'weight': 'bold', 'size': 16})

    # Save Figure
    plt.tight_layout()
    plt.savefig(SAVE_NAME, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    print(f"Plot saved successfully to {SAVE_NAME}")

if __name__ == "__main__":
    df = run_comparison() 
    plot_comparison_refined(df)