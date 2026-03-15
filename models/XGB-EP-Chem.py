#!/usr/bin/env python3
"""
Baseline Model Analysis (XGBoost)
======================================================
Task: Traditional Machine Learning Baseline for Migration Barrier Prediction.
Style: High-impact publication standard (Times New Roman, TIFF).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# ==========================================
# 1. Publication-Quality Plot Settings
# ==========================================
def set_publication_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
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

set_publication_style()

# ==========================================
# 2. Data Loading 
# ==========================================
RANDOM_SEED = 42
TARGET_COL = 'target'

# --- Training Files ---
TRAIN_PATHS = {
    'Ni5Re': 'final_transition_data_Ni5Re_onehot_and_features_relax.csv',
    'Ni8Re': 'final_transition_data_Ni8Re_onehot_and_features_relax.csv',
    'Ni3Re': 'final_transition_data_Ni3Re_onehot_and_features_relax.csv',
    'Ni1Re': 'final_transition_data_Ni1Re_onehot_and_features_relax.csv',
    'Ni10Re': 'final_transition_data_Ni10Re_onehot_and_features_relax.csv',
    'Ni5Al': 'final_transition_data_Ni5Al_onehot_and_features_relax.csv',
    'Ni8Al': 'final_transition_data_Ni8Al_onehot_and_features_relax.csv',
    'Ni3Al': 'final_transition_data_Ni3Al_onehot_and_features_relax.csv',
    'Ni1Al': 'final_transition_data_Ni1Al_onehot_and_features_relax.csv',
    'Ni10Al': 'final_transition_data_Ni10Al_onehot_and_features_relax.csv',
}

# --- Testing Files ---
TEST_PATHS = {
    'Ni10Re': 'final_transition_data_Ni5Al5Re_onehot_and_features_relax.csv'
}

def load_and_clean(path):
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    df = pd.read_csv(path)
    df = df.dropna()
    return df

# 2.1 Load Training Data
print("Loading training datasets...")
train_dfs = []
for key, path in TRAIN_PATHS.items():
    df = load_and_clean(path)
    if df is not None:
        train_dfs.append(df)

if train_dfs:
    df_train = pd.concat(train_dfs, ignore_index=True)
    feature_cols = [c for c in df_train.columns if c != TARGET_COL]
    X_train = df_train[feature_cols]
    y_train = df_train[TARGET_COL].values
    print(f"Total training set shape: {df_train.shape}")
else:
    print("Error: No training data loaded. Generating dummy data for demonstration.")
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
    y_train = np.random.rand(100)
    feature_cols = X_train.columns.tolist()

# 2.2 Load Testing Data
print("Loading testing datasets...")
test_dfs = []
for key, path in TEST_PATHS.items():
    df = load_and_clean(path)
    if df is not None:
        test_dfs.append(df)

if test_dfs:
    df_test = pd.concat(test_dfs, ignore_index=True)
    # Ensure column order consistency
    X_test = df_test[feature_cols] 
    y_test = df_test[TARGET_COL].values
    print(f"Total testing set shape: {df_test.shape}")
else:
    print("Error: No testing data loaded. Generating dummy data for demonstration.")
    X_test = pd.DataFrame(np.random.rand(20, 5), columns=feature_cols)
    y_test = np.random.rand(20)

# ==========================================
# 3. Model Construction & Optimization
# ==========================================
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('xgb', XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1))
])

search_spaces = {
    'xgb__n_estimators': Integer(50, 300),
    'xgb__max_depth': Integer(3, 8),
    'xgb__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
}

print("Starting Bayesian Optimization...")
opt = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_spaces,
    n_iter=20, 
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=RANDOM_SEED
)

try:
    opt.fit(X_train, y_train)
    best_model = opt.best_estimator_
    print("Best parameters found:", opt.best_params_)
except Exception as e:
    print(f"Optimization failed: {e}. Fallback to default model.")
    best_model = pipeline
    best_model.fit(X_train, y_train)

# ==========================================
# 4. Prediction & Evaluation
# ==========================================
y_pred_test = best_model.predict(X_test)

if np.isnan(y_pred_test).sum() > 0:
    print("Warning: Predictions contain NaN, filling with 0.")
    y_pred_test = np.nan_to_num(y_pred_test)

print("\nFinal Evaluation on Test Set:")
print(f"R2: {r2_score(y_test, y_pred_test):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
# ==========================================
# 5. Publication-Quality Plotting Function
# ==========================================
def plot_publication_quality_xgb_final(y_true, y_pred, save_path='parity_plot_XGB_final.tiff'):
    print("Generating Publication-Quality Parity Plot for Baseline...")
    
    # Setup Canvas
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Color Scheme (Deep Teal -> Amber Orange)
    rgb_start = [15, 85, 105]    
    rgb_end   = [230, 120, 40]   
    inset_color_hex = '#0F5569'  
    inset_line_hex  = '#083D4B'
    
    color_start = np.array(rgb_start) / 255
    color_end = np.array(rgb_end) / 255
    custom_cmap = LinearSegmentedColormap.from_list('sophisticated_gradient', [color_start, color_end], N=512)
    
    # Calculate Density
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

    # Plot Scatter
    sc = ax.scatter(x_plot, y_plot, c=z_plot, cmap=cmap, norm=norm, s=45, 
                    alpha=0.9, edgecolor='none', zorder=1) 

    # Reference Line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.1
    lims = [min_val - margin, max_val + margin]
    ax.plot(lims, lims, color='#333333', linestyle='--', linewidth=1.8, zorder=0)
    
    # Axes Setting
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('NEB Calculated Barrier (eV)', fontsize=16, fontweight='medium', color='black') 
    ax.set_ylabel('XGB Predicted Barrier (eV)', fontsize=16, fontweight='medium', color='black') 
    ax.tick_params(direction='in', length=5, width=1.2, labelsize=14, colors='black')

    # Stats Box
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    stats_text = (f"$R^2 = {r2:.3f}$\n" f"MAE = {mae:.3f} eV")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', 
            bbox=dict(boxstyle='square,pad=0.2', facecolor='#F8F9FA', alpha=0.8, edgecolor='none'))

    # Inset Plot
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

    # Colorbar
    if cmap:
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.5)
        
        cbar.set_ticks([]) 
        cbar.minorticks_off()
        cbar.ax.tick_params(which='both', size=0, width=0, length=0)
        
        cbar.ax.text(0.5, -0.03, 'Low', transform=cbar.ax.transAxes, 
                     ha='center', va='top', fontsize=10, color='#444444')
        cbar.ax.text(0.5, 1.03, 'High', transform=cbar.ax.transAxes, 
                     ha='center', va='bottom', fontsize=10, color='#444444')

    plt.tight_layout()
    plt.savefig(save_path, format='tiff', dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()

# ==========================================
# 6. Execute Plotting
# ==========================================
if 'y_test' in locals() and 'y_pred_test' in locals():
    plot_publication_quality_xgb_final(y_test, y_pred_test, save_path='parity_plot_XGB_5Al5Re_com.tiff')
else:
    print("Error: y_test or y_pred_test variables not found. Ensure XGBoost training completed.")