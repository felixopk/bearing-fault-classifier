import matplotlib.pyplot as plt
import os
import pandas as pd

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "features.csv")
PLOTS_DIR = os.path.join(ROOT_DIR, "data", "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("🎨 Visualizing frequency-domain features...")

# Load data
df = pd.read_csv(DATA_PATH)

# Convert to numeric
for col in df.columns:
    if col not in ['label', 'label_original']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute mean features per bearing condition
feature_means = df.groupby('label').mean(numeric_only=True)
print("✅ Computed mean features per bearing type")

# Get Normal bearing and compute difference
normal_features = feature_means.loc['Normal']
comparison = feature_means.subtract(normal_features)

# Select frequency-domain features
freq_features = ['spectral_mean', 'spectral_std', 'spectral_max', 'energy', 'power']

# Create plot
ax = comparison[freq_features].T.plot(kind='bar', figsize=(10,5))
plt.title("Frequency-Domain Feature Differences vs Normal Bearing")
plt.ylabel("Difference from Normal")
plt.xlabel("Frequency-Domain Feature")
plt.grid(True)
plt.tight_layout()

# ✅ Save BEFORE showing
plot_path = os.path.join(PLOTS_DIR, "freq_features.png")
plt.savefig(plot_path)
print(f"✅ Plot saved successfully at: {plot_path}")

# Now show the plot
plt.show()
