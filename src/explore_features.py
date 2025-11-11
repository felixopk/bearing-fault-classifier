"""
01_exploratory_data_analysis.py
Perform detailed Exploratory Data Analysis (EDA) on processed features.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Setup ===
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "features.csv")
PLOTS_DIR = os.path.join(ROOT_DIR, "data", "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.show()

def pause():
    input("\nPress Enter to continue...\n")

# === Header ===
print("python scripts/01_exploratory_data_analysis.py")
print("🚀 Running EDA script...")
print(f"📂 Loading data from: {DATA_PATH}")
print(f"🧠 File exists? {os.path.exists(DATA_PATH)}")

# === Load Data ===
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns\n")

print("="*70)
print("  EXPLORATORY DATA ANALYSIS (EDA)")
print("  Bearing Fault Classification Dataset")
print("="*70, "\n")

# === 1. BASIC DATASET INFORMATION ===
print("="*70)
print("  1. BASIC DATASET INFORMATION")
print("="*70, "\n")

print(f"📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

print("📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col:25s} ({df[col].dtype})")
print()

print("📈 First 5 rows:")
print(df.head(), "\n")

print("📉 Last 5 rows:")
print(df.tail(), "\n")

print("🔢 Data Types:")
print(df.dtypes.value_counts(), "\n")

mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"💾 Memory Usage: {mem:.2f} MB")
pause()

# === 2. CLASS DISTRIBUTION ===
print("="*70)
print("  2. CLASS DISTRIBUTION ANALYSIS")
print("="*70, "\n")

label_col = None
for candidate in ["label", "Label", "class", "fault"]:
    if candidate in df.columns:
        label_col = candidate
        break

if label_col:
    print(f"🏷️  Fault Classes:\n")
    counts = df[label_col].value_counts()
    total = len(df)
    for cls, cnt in counts.items():
        pct = (cnt / total) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cls:15s} [{bar:50s}] {cnt:5d} ({pct:5.1f}%)")

    ratio = counts.max() / counts.min()
    if ratio > 1.5:
        print(f"\n⚠️  Class Imbalance Detected! Ratio (max/min): {ratio:.2f}x")
    else:
        print("\n✅ Class distribution looks balanced!")

    plt.figure()
    sns.countplot(data=df, x=label_col)
    plt.title("Class Distribution")
    plt.savefig(f"{PLOTS_DIR}/01_class_distribution.png", bbox_inches='tight')
    print(f"\n✓ Saved plot: {PLOTS_DIR}/01_class_distribution.png")
    plt.show()
else:
    print("⚠️ No label column found!")

pause()

# === 3. STATISTICAL ANALYSIS ===
print("="*70)
print("  3. STATISTICAL ANALYSIS")
print("="*70, "\n")

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
print(f"📊 Analyzing {len(num_cols)} numeric features\n")
print("📈 Summary Statistics:\n")
print(df[num_cols].describe(), "\n")

missing = df.isnull().sum()
if missing.sum() > 0:
    print("⚠️ Missing values found!")
else:
    print("✓ No missing values found!")

plt.figure(figsize=(14, 8))
df[num_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.savefig(f"{PLOTS_DIR}/02_feature_distributions.png", bbox_inches='tight')
print(f"\n✓ Saved plot: {PLOTS_DIR}/02_feature_distributions.png")
plt.show()

pause()

# === 4. FEATURE COMPARISON BY CLASS ===
print("="*70)
print("  4. FEATURE COMPARISON BY FAULT CLASS")
print("="*70, "\n")

if label_col:
    selected_features = num_cols[:4]
    print(f"📊 Comparing features: {', '.join(selected_features)}\n")
    melted = df.melt(id_vars=label_col, value_vars=selected_features)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x='variable', y='value', hue=label_col)
    plt.title("Feature Distributions by Fault Class")
    plt.savefig(f"{PLOTS_DIR}/03_features_by_class.png", bbox_inches='tight')
    print(f"✓ Saved plot: {PLOTS_DIR}/03_features_by_class.png")
    plt.show()

    print("\n📊 Mean values by class:\n")
    print(df.groupby(label_col)[selected_features].mean(), "\n")

pause()

# === 5. CORRELATION ANALYSIS ===
print("="*70)
print("  5. CORRELATION ANALYSIS")
print("="*70, "\n")

corr = df[num_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.savefig(f"{PLOTS_DIR}/04_correlation_matrix.png", bbox_inches='tight')
print(f"✓ Saved plot: {PLOTS_DIR}/04_correlation_matrix.png")
plt.show()

high_corr = [(i, j, corr.loc[i, j])
             for i in corr.columns for j in corr.columns
             if i != j and abs(corr.loc[i, j]) > 0.8]
if high_corr:
    print(f"\n⚠️ Found {len(high_corr)//2} highly correlated feature pairs (|r| > 0.8):\n")
    for i, j, r in high_corr[:10]:
        print(f"  - {i:20s} ↔ {j:20s} : {r:6.3f}")
else:
    print("✓ No strong correlations found.")
pause()

# === 6. OUTLIER DETECTION ===
print("="*70)
print("  6. OUTLIER DETECTION")
print("="*70, "\n")

outlier_summary = {}
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = ((df[col] < low) | (df[col] > high)).sum()
    outlier_summary[col] = outlier_count

sorted_outliers = sorted(outlier_summary.items(), key=lambda x: x[1], reverse=True)
print("📊 Features with most outliers:\n")
for i, (col, cnt) in enumerate(sorted_outliers[:10], 1):
    pct = cnt / len(df) * 100
    print(f"  {i:2d}. {col:25s}: {cnt:5d} ({pct:5.2f}%)")

plt.figure(figsize=(10, 6))
sns.barplot(x=[c for c, _ in sorted_outliers[:10]],
            y=[cnt for _, cnt in sorted_outliers[:10]])
plt.xticks(rotation=45)
plt.title("Top 10 Features with Most Outliers")
plt.savefig(f"{PLOTS_DIR}/05_outliers.png", bbox_inches='tight')
print(f"\n✓ Saved plot: {PLOTS_DIR}/05_outliers.png")
plt.show()

pause()

# === EDA COMPLETE ===
print("="*70)
print("  EDA COMPLETE!")
print("="*70, "\n")
print(f"✓ All plots saved to: {PLOTS_DIR}/\n")
print("📊 Generated plots:")
for i in range(1, 6):
    print(f"  - {i:02d}_plot.png")
print("\n💡 Next steps:")
print("  1. Review the plots in data/eda_plots/")
print("  2. Train models: python scripts/train_model.py")
print("  3. Create Jupyter notebook for deeper analysis\n")
