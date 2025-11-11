"""
Explore the processed features dataset
"""

import pandas as pd
import sys
import os
print("🚀 Running EDA script...")


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(ROOT_DIR, "data", "processed", "features.csv")

print(f"📂 Loading data from: {data_path}")
print(f"🧠 File exists? {os.path.exists(data_path)}")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
else:
    print("❌ File not found.")

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

print("=" * 70)
print("  EXPLORING PROCESSED FEATURES")
print("=" * 70)
print()

try:
    # Load the processed features
    print("📂 Loading data/processed/features.csv...")
    df = pd.read_csv("data/processed/features.csv")
    print("✅ Dataset loaded successfully!")
    print()
    
    # Basic info
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print()
    print(f"📊 Shape: {df.shape[0]} samples × {df.shape[1]} features")
    print()
    
    # Column names
    print("📋 Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    print()
    
    # Check for label column
    label_col = None
    for possible_label in ['label', 'Label', 'fault', 'Fault', 'class', 'Class']:
        if possible_label in df.columns:
            label_col = possible_label
            break
    
    if label_col:
        print(f"🏷️  Label column: '{label_col}'")
        print()
        print("Class distribution:")
        label_counts = df[label_col].value_counts()
        total = len(df)
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {str(label):20s} [{bar:50s}] {count:5d} ({percentage:5.1f}%)")
        print()
    else:
        print("⚠️  No label column found!")
        print()
    
    # First few rows
    print("=" * 70)
    print("SAMPLE DATA (first 5 rows)")
    print("=" * 70)
    print()
    
    # Show first 5 rows but limit columns for readability
    display_cols = df.columns[:10].tolist()
    if label_col and label_col not in display_cols:
        display_cols.append(label_col)
    
    print(df[display_cols].head())
    print()
    
    if len(df.columns) > 10:
        print(f"... and {len(df.columns) - len(display_cols)} more columns")
        print()
    
    # Statistics
    print("=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print()
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) > 0:
        print("Basic statistics (first 5 numeric features):")
        print()
        stats_df = df[numeric_cols[:5]].describe()
        print(stats_df)
        print()
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("⚠️  Missing values:")
        for col, count in missing[missing > 0].items():
            print(f"  - {col}: {count} ({count/len(df)*100:.1f}%)")
        print()
    else:
        print("✅ No missing values!")
        print()
    
    print("=" * 70)
    print("EXPLORATION COMPLETE")
    print("=" * 70)
    print()
    print("💡 Next steps:")
    print("  1. Train models: python scripts/train_model.py")
    print("  2. Visualize in Jupyter: jupyter notebook notebooks/")
    print()
    
    sys.stdout.flush()

except FileNotFoundError:
    print("❌ Error: data/processed/features.csv not found!")
    print()
    print("💡 Generate it first:")
    print("   python scripts/prepare_npz_cnn_data.py")
    print()
    sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)