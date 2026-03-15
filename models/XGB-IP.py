#!/usr/bin/env python3
"""
Baseline Model Analysis (XGBoost - Single File 80/20 Split)
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
from scipy.stats import gaussian_kde, norm

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
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
# 2. Configuration & Data Loading
# ==========================================
RANDOM_SEED = 42
DATA_PATH = 'final_transition_data_Ni5Al_onehot_and_features_relax.csv' 
TARGET_COL = 'target'

def load_and_split_data(path, target_col, test_size=0.2):
    if not os.path.exists(path):
        print(f"Warning: File not found at {path}")
        print("Using dummy data for demonstration.")
        df = pd.DataFrame(np.random.rand(200, 6), columns=[f'f{i}' for i in range(5)] + [target_col])
        df[target_col] = 3 * df['f0'] + 2 * df['f1'] + np.random.normal(0, 0.1, 200)
    else:
        df = pd.read_csv(path)
        df = df.dropna()
        print(f"Successfully loaded data. Shape: {df.shape}")

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, shuffle=True
    )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH, TARGET_COL)

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

print("\nFinal Evaluation on Test Set:")
print(f"R2: {r2_score(y_test, y_pred_test):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")

# ==========================================
# 5. Publication-Quality Plotting Function
# ==========================================
def plot_publication_quality_parity(y_true, y_pred, save_path='parity_plot_5Al.tiff'):
    print("Generating Publication-Quality Parity Plot...")
    
    # Setup Canvas
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Calculate Density
    xy = np.vstack([y_true, y_pred])
    try:
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_plot, y_plot, z_plot = y_true[idx], y_pred[idx], z[idx]
        cmap = 'viridis' 
    except:
        x_plot, y_plot, z_plot = y_true, y_pred, 'blue'
        cmap = None

    # Plot Scatter
    sc = ax.scatter(x_plot, y_plot, c=z_plot, cmap=cmap, s=40, 
                    alpha=0.9, edgecolor='none', zorder=1)

    # Reference Line (Gray Dashed)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.1
    lims = [min_val - margin, max_val + margin]
    
    ax.plot(lims, lims, color='gray', linestyle='--', linewidth=1.5, zorder=0, label='Ideal')
    
    # Axes Setting
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('NEB Calculated Barrier (eV)', fontsize=16) 
    ax.set_ylabel('XGB Predicted Barrier (eV)', fontsize=16)
    ax.tick_params(direction='in', length=5, width=1.0, labelsize=14)

    # Stats Box
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    stats_text = (
        f"$R^2 = {r2:.3f}$\n"
        f"MAE = {mae:.3f} eV"
    )
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', 
            bbox=dict(boxstyle='square,pad=0.4', facecolor='#E8F5E9', alpha=0.9, edgecolor='none'))

    # Inset Plot (Transparent Background)
    ax_inset = ax.inset_axes([0.59, 0.08, 0.38, 0.30]) 
    ax_inset.set_zorder(10)
    
    residuals = y_true - y_pred
    
    sns.histplot(residuals, ax=ax_inset, kde=True, 
                 color='#333333', stat='density', 
                 element="step", alpha=0.3, linewidth=0)
    sns.kdeplot(residuals, ax=ax_inset, color='black', linewidth=1.5)
    ax_inset.axvline(0, color='red', linestyle='--', linewidth=1.0)
    
    ax_inset.set_xlabel('Residuals (eV)', fontsize=11, labelpad=1) 
    ax_inset.set_ylabel('')
    ax_inset.set_yticks([])
    ax_inset.tick_params(axis='x', labelsize=9, direction='in', pad=2) 
    
    ax_inset.patch.set_alpha(0.0) 
    
    for spine in ax_inset.spines.values():
        spine.set_linestyle('--')   
        spine.set_linewidth(1.0)    
        spine.set_color('#555555') 

    # Colorbar
    if cmap:
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.8)
        cbar.set_ticks([]) 
        cbar.set_label('Point Density', rotation=270, labelpad=15, fontsize=12)
        cbar.ax.text(0.5, -0.02, 'Low', transform=cbar.ax.transAxes, 
                     ha='center', va='top', fontsize=10)
        cbar.ax.text(0.5, 1.02, 'High', transform=cbar.ax.transAxes, 
                     ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, format='tiff', dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()

# ==========================================
# 6. Execute Plotting
# ==========================================
plot_publication_quality_parity(y_test, y_pred_test, save_path='parity_plot_5Al.tiff')