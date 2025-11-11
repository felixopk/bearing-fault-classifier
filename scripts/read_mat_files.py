import os
import scipy.io as sio
import pandas as pd

RAW_DIR = "data/raw/processed_raw"
OUT_PATH = "data/processed/raw_data.csv"

def read_mat_files():
    all_data = []

    for file_name in os.listdir(RAW_DIR):
        if file_name.endswith(".mat"):
            file_path = os.path.join(RAW_DIR, file_name)
            print(f"📂 Reading {file_name}...")

            mat_data = sio.loadmat(file_path)

            # Get the actual signal key
            signal_keys = [k for k in mat_data.keys() if not k.startswith("__")]
            if not signal_keys:
                print(f"⚠️ No valid keys found in {file_name}, skipping...")
                continue

            key = signal_keys[0]
            signal = mat_data[key]

            # ✅ Ensure signal is 1D
            signal = signal.flatten()

            # Create DataFrame for this file
            df = pd.DataFrame({
                "value": signal,
                "source_file": file_name
            })

            all_data.append(df)

    if not all_data:
        print("❌ No .mat files found. Check your data/raw/processed_raw folder.")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def main():
    os.makedirs("data/processed", exist_ok=True)
    df = read_mat_files()

    if df is None:
        return

    # Create a simple label based on file name
    df["label"] = df["source_file"].apply(
        lambda x: (
            "ball" if "B0" in x else
            "inner_race" if "IR" in x else
            "outer_race" if "OR" in x else
            "normal"
        )
    )

    df.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Saved processed data to: {OUT_PATH}")
    print(f"📊 Shape: {df.shape}")
    print(f"📄 Columns: {list(df.columns)}")
    print(df.head())


if __name__ == "__main__":
    main()
