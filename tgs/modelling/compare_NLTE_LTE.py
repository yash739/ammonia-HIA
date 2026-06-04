import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_LTE_CSV = "/home/yasho379/magritte_rebuilt/output_test_1e-6_LTE/results/NLTE_nh3_1e-6_LTE.csv"
DEFAULT_NLTE_CSV = "/home/yasho379/magritte_rebuilt/output_test_1e-6_parallel_12rays_v2/results/NLTE_nh3_1e-6_parallel_12rays_v2.csv"
DEFAULT_COLUMNS = [
    "A_10",
    "A_21",
    "A_MAIN",
    "A_12",
    "A_01",
    "R_01_MAIN",
    "R_10_MAIN",
    "R_21_MAIN",
    "R_12_MAIN",
    "Main Hyperfine Optical Depth",
]


def load_dataframe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def validate_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label} data: {missing}")


def compute_ratios(lte_df, nlte_df, columns):
    # Keys used to identify matching model rows between LTE and NLTE files
    keys = ["T_cloud", "vturb", "XNH3", "numberdensity", "radius_req", "N_NH3"]

    # Validate key columns exist
    validate_columns(lte_df, keys, "LTE")
    validate_columns(nlte_df, keys, "NLTE")

    # Validate requested value columns exist in both
    validate_columns(lte_df, columns, "LTE")
    validate_columns(nlte_df, columns, "NLTE")

    # Merge on keys to find matching combinations
    merged = pd.merge(lte_df, nlte_df, on=keys, how="inner", suffixes=("_lte", "_nlte"))
    if merged.empty:
        raise ValueError("No matching rows found between LTE and NLTE files using the specified key columns.")

    if len(merged) != len(lte_df) or len(merged) != len(nlte_df):
        print(f"Warning: Number of matched rows is {len(merged)} (LTE: {len(lte_df)}, NLTE: {len(nlte_df)}). Using matched rows only.")

    # Build the ratio DataFrame starting with the matching keys
    ratio_df = merged[keys].copy()

    # Compute ratios for each requested column using the merged table
    for col in columns:
        lcol = f"{col}_lte"
        ncol = f"{col}_nlte"
        if lcol not in merged.columns or ncol not in merged.columns:
            raise ValueError(f"Column {col} not found in merged LTE/NLTE data after suffixing.")
        # Convert to float and avoid division by zero
        numer = merged[ncol].astype(float)
        denom = merged[lcol].astype(float)
        with pd.option_context('mode.use_inf_as_na', True):
            ratio_df[col] = numer - denom

    return ratio_df


def save_ratio_dataframe(ratio_df, output_path):
    output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    csv_path = os.path.join(output_dir, "ratio_values.csv")
    ratio_df.to_csv(csv_path, index=False)
    print(f"Saved ratio data CSV to: {csv_path}")


def plot_ratio_histograms(ratio_df, output_path):
    cols = [c for c in ratio_df.columns if c not in ["T_cloud", "vturb", "XNH3", "numberdensity", "radius_req"]]
    ncols = min(3, len(cols))
    nrows = (len(cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]
    for ax, col in zip(axes_flat, cols):
        values = ratio_df[col].dropna()
        ax.hist(values, bins=200, alpha=0.8, edgecolor="black")
        ax.set_title(f"NLTE / LTE ratio: {col}")
        ax.set_xlabel("Ratio")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.3)
        # ax.set_xlim(0, 2)  # Set x-axis limits for better visualization

    for ax in axes_flat[len(cols):]:
        fig.delaxes(ax)

    fig.suptitle("Histogram of NLTE / LTE ratios", fontsize=16)
    fig.savefig(output_path, dpi=150)
    print(f"Saved histogram plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare NLTE and LTE CSV files and plot ratio histograms.")
    parser.add_argument("--lte-csv", default=DEFAULT_LTE_CSV, help="Path to the LTE CSV file.")
    parser.add_argument("--nlte-csv", default=DEFAULT_NLTE_CSV, help="Path to the NLTE CSV file.")
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS, help="Column names to compare and plot ratios for.")
    parser.add_argument("--output", default="ratio_histograms.png", help="Output path for the ratio histogram plot.")
    args = parser.parse_args()

    lte_df = load_dataframe(args.lte_csv)
    nlte_df = load_dataframe(args.nlte_csv)
    ratio_df = compute_ratios(lte_df, nlte_df, args.columns)
    save_ratio_dataframe(ratio_df, args.output)
    plot_ratio_histograms(ratio_df, args.output)


if __name__ == "__main__":
    main()
