import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("🚀 Running BEARING_features script...")

# === Setup ===
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "features.csv")
PLOTS_DIR = os.path.join(ROOT_DIR, "data", "bearing_features")
os.makedirs(PLOTS_DIR, exist_ok=True)

#--------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(df.head)
print(df.columns)
print(df['label'].value_counts())
print(df['label_original'].value_counts)

#This helps you summarize and compare average feature behavior per condition.
for col in df.columns:
    if col not in ['label', 'label_original']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=[c for c in df.columns if c not in ['label', 'label_original']])
feature_means = df.groupby('label').mean(numeric_only=True)
print(feature_means)

#Step 4: Compare Normal vs Faulty Bearings
#This gives how much each fault deviates from the normal bearing for each feature.
normal_features = feature_means.loc['Normal']
comparison = feature_means.subtract(normal_features)
print(comparison)
comparison.T.plot(kind='bar', figsize=(12,6))
plt.title("Feature Differences vs Normal Bearing")
plt.ylabel("Difference from Normal")
plt.xlabel("Feature")
plt.show()




#Radar chart (spider plot) for one or two cases
#Radar plots are great for visualizing feature patterns.
labels = feature_means.columns
num_features = len(labels)

normal = feature_means.loc['Normal'].values
ball = feature_means.loc['Ball'].values

angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
normal = np.concatenate((normal,[normal[0]]))
ball = np.concatenate((ball,[ball[0]]))
angles += angles[:1]

plt.figure(figsize=(6,6))
plt.polar(angles, normal, label='Normal')
plt.polar(angles, ball, label='Ball')
plt.fill(angles, normal, alpha=0.1)
plt.fill(angles, ball, alpha=0.1)
plt.title("Radar Comparison: Normal vs Ball")
plt.legend()