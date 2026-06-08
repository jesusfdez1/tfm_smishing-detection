import argparse
import sys
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Extract a customized subset from the MetaSMS master dataset.")
    parser.add_argument("--input", type=str, default="data/processed/metasms_hss_master.csv", help="Path to the master CSV dataset.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the customized subset CSV.")
    
    # Data filters
    parser.add_argument("--labels", type=str, nargs="+", choices=["spam", "ham", "smishing"], help="Filter by specific labels (e.g., --labels spam smishing).")
    parser.add_argument("--languages", type=str, nargs="+", help="Filter by languages (e.g., --languages en es).")
    parser.add_argument("--sources", type=str, nargs="+", help="Filter by specific sources.")
    
    # AI filters
    parser.add_argument("--exclude-ai", action="store_true", help="Remove all AI-generated messages.")
    parser.add_argument("--only-ai", action="store_true", help="Keep ONLY AI-generated messages.")
    
    # Length filters
    parser.add_argument("--min-length", type=int, help="Minimum character length of the message.")
    parser.add_argument("--max-length", type=int, help="Maximum character length of the message.")
    
    # Sampling
    parser.add_argument("--sample", type=int, help="Randomly sample exactly N rows. It maintains the class proportion (stratified) if possible.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # Columns
    parser.add_argument("--keep-cols", type=str, nargs="+", help="Specific columns to keep. Example: --keep-cols text_anonymized canonical_label")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    print(f"Loading master dataset: {input_path}...")
    # Read everything as string to avoid type issues, except is_ai_generated which is useful as int
    df = pd.read_csv(input_path, dtype=str)
    
    if "is_ai_generated" in df.columns:
        df["is_ai_generated"] = pd.to_numeric(df["is_ai_generated"], errors="coerce").fillna(0).astype(int)

    initial_len = len(df)
    print(f"Loaded {initial_len} rows.")

    # 1. Filter Labels
    if args.labels:
        df = df[df["canonical_label"].isin(args.labels)]
        print(f"Filtered by labels {args.labels}. Remaining: {len(df)}")

    # 2. Filter Languages
    if args.languages:
        df = df[df["language"].isin(args.languages)]
        print(f"Filtered by languages {args.languages}. Remaining: {len(df)}")

    # 3. Filter Sources
    if args.sources:
        df = df[df["source"].isin(args.sources)]
        print(f"Filtered by sources {args.sources}. Remaining: {len(df)}")

    # 4. Filter AI
    if args.exclude_ai:
        df = df[df.get("is_ai_generated", 0) == 0]
        print(f"Excluded AI generated. Remaining: {len(df)}")
    elif args.only_ai:
        df = df[df.get("is_ai_generated", 0) == 1]
        print(f"Kept ONLY AI generated. Remaining: {len(df)}")

    # 5. Filter Length
    if args.min_length or args.max_length:
        text_len = df["text"].fillna("").str.len()
        if args.min_length:
            df = df[text_len >= args.min_length]
            print(f"Applied min_length >= {args.min_length}. Remaining: {len(df)}")
        if args.max_length:
            df = df[text_len <= args.max_length]
            print(f"Applied max_length <= {args.max_length}. Remaining: {len(df)}")

    # 6. Stratified Sampling
    if args.sample and args.sample < len(df):
        print(f"Stratified sampling to {args.sample} rows...")
        try:
            original_df = df
            # Attempt stratified sampling using groupby
            df = original_df.groupby("canonical_label", group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), int(args.sample * len(x) / len(original_df))), random_state=args.seed)
            )
            # If rounding leaves us with fewer rows, fill with a random sample of the remainder
            shortfall = args.sample - len(df)
            if shortfall > 0:
                remaining = original_df[~original_df.index.isin(df.index)].sample(n=shortfall, random_state=args.seed)
                df = pd.concat([df, remaining])
        except Exception as e:
            print("Stratified sampling failed (maybe missing classes), falling back to random sampling.")
            # If original_df exists in locals, use it to fallback, otherwise use df.
            fallback_df = locals().get('original_df', df)
            df = fallback_df.sample(n=args.sample, random_state=args.seed)
            
        print(f"Sampled exactly {len(df)} rows.")

    # 7. Column Selection
    if args.keep_cols:
        missing_cols = [c for c in args.keep_cols if c not in df.columns]
        if missing_cols:
            print(f"Warning: Columns not found in dataset and will be ignored: {missing_cols}")
        valid_cols = [c for c in args.keep_cols if c in df.columns]
        df = df[valid_cols]
        print(f"Kept columns: {valid_cols}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print("\n--- FINAL DATASET SUMMARY ---")
    print(f"Total rows: {len(df)}")
    if "canonical_label" in df.columns:
        print("\nLabel Distribution:")
        print(df["canonical_label"].value_counts().to_string())
    print(f"\nSaved subset to: {output_path}")

if __name__ == "__main__":
    main()
