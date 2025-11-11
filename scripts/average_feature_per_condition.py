import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
print("🚀 Running BEARING_feature per condition script...")

# === Setup ===
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "features.csv")
PLOTS_DIR = os.path.join(ROOT_DIR, "data", "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("compute the average feature per condition")
df = pd.read_csv(DATA_PATH)
for col in df.columns:
    if col not in ['label', 'label_original']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
#Compute mean features per bearing type
print("computing features per bearing type.....")

feature_means = df.groupby('label').mean(numeric_only=True)
print(feature_means)

#📊 Step 3 — Compare all faults to Normal
print("comparing all fault to normal bearing ..........")
normal = feature_means.loc['Normal']
comparison = feature_means.subtract(normal)
comparison
# Compute difference from Normal
normal_features = feature_means.loc['Normal']
comparison = feature_means.subtract(normal_features)

print("\n📊 Difference from Normal bearing:")
print(comparison)



comparison.T.plot(kind='bar', figsize=(12,6))
plt.title("Feature Difference vs Normal Bearing")
plt.ylabel("Change Compared to Normal")
plt.xlabel("Feature")
plt.grid(True)
# ✅ Save BEFORE showing
plt_path = os.path.join(PLOTS_DIR,"feature_differences_to_normal.png")
plt.savefig(plt_path)
plt.show()
