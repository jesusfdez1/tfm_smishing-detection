import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Create balanced, frozen ML splits (train/val/test) from the master dataset.")
    parser.add_argument("--input", type=str, default="data/processed/metasms_v1_enriched.csv", help="Path to the master CSV dataset.")
    parser.add_argument("--output-dir", type=str, default="data/processed/model_splits", help="Directory to save the splits.")
    parser.add_argument("--max-ham-per-source", type=int, default=20000, help="Maximum number of ham messages to keep per source dataset to prevent source bias.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading master dataset: {input_path}...")
    df = pd.read_csv(input_path, dtype=str)
    
    initial_len = len(df)
    print(f"Loaded {initial_len} rows.")

    # 1. Filter out rows without canonical_label
    df = df[df["canonical_label"].isin(["spam", "ham", "smishing"])]
    print(f"Rows with valid canonical_label: {len(df)}")

    # 2. Source Bias Mitigation for 'ham'
    print(f"\n--- Mitigating Ham Source Bias (Max {args.max_ham_per_source} per source) ---")
    
    spam_smish_df = df[df["canonical_label"].isin(["spam", "smishing"])]
    ham_df = df[df["canonical_label"] == "ham"]
    
    # Group by source and sample
    sampled_hams = []
    for source, group in ham_df.groupby("reference"):
        n_rows = len(group)
        if n_rows > args.max_ham_per_source:
            sampled = group.sample(n=args.max_ham_per_source, random_state=args.seed)
            print(f"  Source '{source}': reduced from {n_rows} to {args.max_ham_per_source}")
            sampled_hams.append(sampled)
        else:
            print(f"  Source '{source}': kept all {n_rows}")
            sampled_hams.append(group)
            
    if sampled_hams:
        ham_balanced = pd.concat(sampled_hams)
    else:
        ham_balanced = pd.DataFrame(columns=df.columns)

    final_df = pd.concat([spam_smish_df, ham_balanced]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    print("\n--- Balanced Dataset Distribution ---")
    print(final_df["canonical_label"].value_counts())
    
    # 3. Stratified Split 80/10/10
    print("\n--- Performing 80/10/10 Stratified Split ---")
    # First split: 80% Train, 20% Temp
    train_df, temp_df = train_test_split(
        final_df, 
        test_size=0.20, 
        stratify=final_df["canonical_label"], 
        random_state=args.seed
    )
    
    # Second split: 50% of Temp -> 10% Val, 10% Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df["canonical_label"], 
        random_state=args.seed
    )

    print(f"Train size: {len(train_df)} ({len(train_df)/len(final_df):.1%})")
    print(f"Val size:   {len(val_df)} ({len(val_df)/len(final_df):.1%})")
    print(f"Test size:  {len(test_df)} ({len(test_df)/len(final_df):.1%})")

    # 4. Save splits
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    print(f"\nSaving to {output_dir}...")
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    print("Done!")

if __name__ == "__main__":
    main()
