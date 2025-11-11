"""
Inspect existing CWRU dataset files to understand their structure
"""

import numpy as np
import os
from pathlib import Path


def inspect_npz_file(filepath):
    """Inspect an .npz file"""
    
    print(f"\n{'='*70}")
    print(f"Inspecting: {filepath}")
    print('='*70)
    
    try:
        data = np.load(filepath, allow_pickle=True)
        
        print(f"\n📊 Keys in file: {list(data.keys())}")
        print()
        
        for key in data.keys():
            value = data[key]
            print(f"Key: '{key}'")
            print(f"  Type: {type(value)}")
            
            if hasattr(value, 'shape'):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                
                # Show some statistics
                if value.dtype in [np.float32, np.float64, np.int32, np.int64]:
                    try:
                        if len(value.shape) == 1:
                            print(f"  Length: {len(value)}")
                            print(f"  Min: {value.min():.4f}, Max: {value.max():.4f}")
                            print(f"  Mean: {value.mean():.4f}, Std: {value.std():.4f}")
                            print(f"  First 5 values: {value[:5]}")
                        elif len(value.shape) == 2:
                            print(f"  Rows: {value.shape[0]}, Cols: {value.shape[1]}")
                            print(f"  Sample row 0: {value[0][:10]}...")
                        elif len(value.shape) == 3:
                            print(f"  3D array: {value.shape}")
                            print(f"  Sample [0,0,:5]: {value[0, 0, :5]}")
                    except:
                        pass
            else:
                print(f"  Value: {value}")
            
            print()
        
        data.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def inspect_csv_file(filepath):
    """Inspect a CSV file"""
    
    print(f"\n{'='*70}")
    print(f"Inspecting: {filepath}")
    print('='*70)
    
    try:
        import pandas as pd
        df = pd.read_csv(filepath, nrows=5)
        
        print(f"\n📊 CSV Shape: {df.shape}")
        print(f"Columns ({len(df.columns)}): {list(df.columns)[:10]}")
        print()
        print("First 3 rows:")
        print(df.head(3))
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main inspection function"""
    
    print("=" * 70)
    print("  INSPECTING EXISTING CWRU DATASET FILES")
    print("=" * 70)
    
    raw_dir = Path('data/raw')
    
    # Find all data files
    npz_files = list(raw_dir.glob('*.npz'))
    csv_files = list(raw_dir.glob('*.csv'))
    
    print(f"\nFound files:")
    print(f"  - .npz files: {len(npz_files)}")
    print(f"  - .csv files: {len(csv_files)}")
    print()
    
    # Inspect NPZ files
    for npz_file in npz_files:
        inspect_npz_file(npz_file)
    
    # Inspect CSV files  
    for csv_file in csv_files:
        inspect_csv_file(csv_file)
    
    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)
    print()
    print("💡 Based on the file structure, we can:")
    print("   1. Use the CNN data file directly if it has labeled data")
    print("   2. Use the feature CSV if features are already extracted")
    print("   3. Download fresh dataset if needed")
    print()


if __name__ == "__main__":
    main()
