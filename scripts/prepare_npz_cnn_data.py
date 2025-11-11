"""
Prepare CNN NPZ data for classical ML training
Extracts features from the 32x32 spectrograms
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))




def prepare_npz_cnn_data():
    """
    Load NPZ CNN data and extract features from time-series
    The data is in 32x32 spectrogram format - we'll flatten or extract features
    """
    
    print("=" * 70)
    print("  PREPARING NPZ CNN DATA FOR TRAINING")
    print("=" * 70)
    print()
    
    # Load data
    npz_file = 'data/raw/CWRU_48k_load_1_CNN_data.npz'
    
    print(f"📂 Loading: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    
    X = data['data']      # Shape: (4600, 32, 32) - spectrograms
    y = data['labels']    # Shape: (4600,) - labels
    
    print(f"✓ Loaded {len(X)} samples")
    print(f"  - Data shape: {X.shape}")
    print(f"  - Labels shape: {y.shape}")
    print()
    
    # Check unique labels
    unique_labels = np.unique(y)
    print(f"🏷️  Unique labels ({len(unique_labels)}):")
    for label in unique_labels:
        count = np.sum(y == label)
        print(f"  - {label:20s}: {count:5d} samples ({count/len(y)*100:.1f}%)")
    print()
    
    # Option 1: Flatten spectrograms to use as features (simple but many features)
    print("📊 Preparing features...")
    print("  Method: Flattening 32x32 spectrograms + statistical features")
    print()
    
    all_features = []

    
    for i, (sample, label) in enumerate(zip(X, y)):
        if i % 500 == 0:
            print(f"  Processing: {i}/{len(X)} samples...", end='\r')
        
        # Flatten the 32x32 spectrogram
        flattened = sample.flatten()
        
        # Extract statistical features from the flattened data
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(flattened)
        features['std'] = np.std(flattened)
        features['max'] = np.max(flattened)
        features['min'] = np.min(flattened)
        features['median'] = np.median(flattened)
        
        # RMS
        features['rms'] = np.sqrt(np.mean(flattened**2))
        
        # Shape features
        from scipy import stats
        features['kurtosis'] = stats.kurtosis(flattened)
        features['skewness'] = stats.skew(flattened)
        
        # Percentiles
        features['p25'] = np.percentile(flattened, 25)
        features['p75'] = np.percentile(flattened, 75)
        features['iqr'] = features['p75'] - features['p25']
        
        # Energy
        features['energy'] = np.sum(flattened**2)
        features['power'] = features['energy'] / len(flattened)
        
        # Spectral features (from rows - frequency bands)
        features['spectral_mean'] = np.mean(np.mean(sample, axis=1))
        features['spectral_std'] = np.std(np.mean(sample, axis=1))
        features['spectral_max'] = np.max(np.mean(sample, axis=1))
        
        # Temporal features (from columns - time steps)
        features['temporal_mean'] = np.mean(np.mean(sample, axis=0))
        features['temporal_std'] = np.std(np.mean(sample, axis=0))
        features['temporal_max'] = np.max(np.mean(sample, axis=0))
        
        # Add label
        features['label'] = str(label)
        
        all_features.append(features)
    
    print(f"  Processing: {len(X)}/{len(X)} samples... Done!     ")
    print()
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Standardize labels
    label_mapping = {
        'Normal_0': 'Normal',
        'Normal_1': 'Normal',
        'Normal_2': 'Normal',
        'Normal_3': 'Normal',
        'IR_007_0': 'Inner_Race',
        'IR_007_1': 'Inner_Race',
        'IR_007_2': 'Inner_Race',
        'IR_007_3': 'Inner_Race',
        'Ball_007_0': 'Ball',
        'Ball_007_1': 'Ball',
        'Ball_007_2': 'Ball',
        'Ball_007_3': 'Ball',
        'OR_007_6_0': 'Outer_Race',
        'OR_007_6_1': 'Outer_Race',
        'OR_007_6_2': 'Outer_Race',
        'OR_007_6_3': 'Outer_Race',
    }
    
    # Map or use generic mapping
    df['label_mapped'] = df['label'].map(label_mapping)
    
    if df['label_mapped'].isna().any():
        def generic_mapping(label):
            label = str(label).upper()
            if 'NORMAL' in label:
                return 'Normal'
            elif 'IR' in label or 'INNER' in label:
                return 'Inner_Race'
            elif 'BALL' in label or 'B_' in label or '_B_' in label:
                return 'Ball'
            elif 'OR' in label or 'OUTER' in label:
                return 'Outer_Race'
            else:
                return 'Unknown'
        
        df['label_mapped'] = df['label'].apply(generic_mapping)
    
    # Replace label column
    df['label_original'] = df['label']
    df['label'] = df['label_mapped']
    df = df.drop('label_mapped', axis=1)
    
    print("🏷️  Final label distribution:")
    for label, count in df['label'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  - {label:20s}: {count:5d} samples ({percentage:.1f}%)")
    print()
    
    # Save
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'features.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved to: {output_file}")
    print()
    
    # Summary
    print("=" * 70)
    print("  PREPARATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"📊 Final Dataset:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Features: {len(df.columns) - 2}")  # Exclude label columns
    print(f"  - Classes: {df['label'].nunique()}")
    print()
    
    print("💡 Next steps:")
    print("  1. Train models: python scripts/train_model.py")
    print("  2. Explore data: jupyter notebook")
    print()
    
    return df


if __name__ == "__main__":
    try:
        df = prepare_npz_cnn_data()
        
        # Show sample
        print("📋 Sample data:")
        print("=" * 70)
        cols_to_show = ['mean', 'std', 'rms', 'kurtosis', 'skewness', 'label']
        print(df[cols_to_show].head(3))
        print()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 Make sure you're in the project root directory:")
        print(f"   cd ~/DevOps/Bearing-Fault-Classifier")
        print(f"   python scripts/prepare_npz_cnn_data.py")