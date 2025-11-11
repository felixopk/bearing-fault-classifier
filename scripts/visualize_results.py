"""Generate visualizations for model results"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import os 


# === Setup ===
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "features.csv")
PLOTS_DIR = os.path.join(ROOT_DIR, "data", "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
df = pd.read_csv('data/processed/features.csv')
feature_cols = [col for col in df.columns if col not in ['label', 'label_original', 'fault']]
X = df[feature_cols].values
y = df['label'].values

# Load models
models_dir = Path('models')
rf_model = joblib.load(models_dir / 'random_forest_model.pkl')
rf_scaler = joblib.load(models_dir / 'random_forest_scaler.pkl')

# Scale and predict
X_scaled = rf_scaler.transform(X)
y_pred = rf_model.predict(X_scaled)

# Create confusion matrix
cm = confusion_matrix(y, y_pred, labels=['Ball', 'Inner_Race', 'Normal', 'Outer_Race'])

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ball', 'Inner_Race', 'Normal', 'Outer_Race'],
            yticklabels=['Ball', 'Inner_Race', 'Normal', 'Outer_Race'],
            cbar_kws={'label': 'Count'})
plt.title('Random Forest - Confusion Matrix\nAccuracy: 96.20%', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

# Save
output_dir = Path('data/eda_plots')
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion matrix to data/eda_plots/confusion_matrix_rf.png")
plt.show()

# Model comparison
results = {
    'Model': ['Random Forest', 'SVM', 'XGBoost', 'Logistic Regression'],
    'Accuracy': [96.20, 96.09, 95.98, 89.46],
    'F1-Score': [96.19, 96.07, 95.97, 89.36]
}

df_results = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
x = np.arange(len(df_results))
width = 0.35

plt.bar(x - width/2, df_results['Accuracy'], width, label='Accuracy', color='#2ecc71')
plt.bar(x + width/2, df_results['F1-Score'], width, label='F1-Score', color='#3498db')

plt.xlabel('Model', fontsize=12)
plt.ylabel('Score (%)', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, df_results['Model'], rotation=15, ha='right')
plt.legend()
plt.ylim([85, 100])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved comparison plot to data/eda_plots/model_comparison.png")
plt.show()