import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import seaborn as sns
import matplotlib.pyplot as plt

def create_comparison_plot(df: pd.DataFrame, our_col: str, ihs_col: str, plot_title: str, output_path: str) -> None:
    """
    Create correlation plots comparing our calculations with IHS values
    
    Args:
        df: DataFrame containing both columns to compare
        our_col: Name of column containing our calculations
        ihs_col: Name of column containing IHS values
        plot_title: Title for the plot
        output_path: Where to save the plot
    """
    # Removing one outlier
    df = df[df[our_col] < 0.25e7]
    pearson_corr = df[our_col].corr(df[ihs_col], method='pearson')
    spearman_corr = df[our_col].corr(df[ihs_col], method='spearman')

    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")    
    
    plt.figure(figsize=(10, 6))
    
    max_val = max(df[our_col].max(), df[ihs_col].max())
    
    sns.scatterplot(x=our_col, y=ihs_col, data=df, alpha=0.5)

    x = df[our_col]
    y = df[ihs_col]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8, label='Line of best fit')
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect correlation (x=y)')

    plt.title(f"{plot_title}\nPearson correlation: {pearson_corr:.4f}\nSpearman correlation: {spearman_corr:.4f}")
    plt.xlabel(f"{our_col} (Our estimate)")
    plt.ylabel(f"{ihs_col} (IHS-V)")
    plt.legend()
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def read_files(config: dict) -> pd.DataFrame:
    input_dir = Path(config["input_dir"])
    csv_files = list(input_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in input directory")
    merged_df = None
    for file in csv_files:
        df = pd.read_csv(file, index_col=0)
        if 'case_id' in df.columns:
            df = df.drop('case_id', axis=1)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on='HHID', how='outer')
    consumption_agg_df = pd.read_stata(input_dir / "ihs5_consumption_aggregate.dta", convert_categoricals=False)
    consumption_agg_df = consumption_agg_df[["HHID", "price_indexL", "rexpagg", "expagg", "poor"]]
    merged_df = merged_df.merge(consumption_agg_df, how="left", on="HHID")
    return merged_df

def main(config: dict) -> None:
    df = read_files(config)
    df["Total consumption (annual) (nominal)"] = df[[x for x in df.columns if "Consumption (annual) (nominal)" in x]].sum(axis=1)
    df["Total consumption (annual) (real)"] = df[[x for x in df.columns if "Consumption (annual) (real)" in x]].sum(axis=1)
    create_comparison_plot(
        df,
        "Total consumption (annual) (nominal)",
        "expagg",
        "Total consumption (nominal) vs expagg",
        Path("plots") / "total_consumption_nominal_vs_expagg.png"
    )
    
    create_comparison_plot(
        df,
        "Total consumption (annual) (real)",
        "rexpagg",
        "Total consumption (real) vs rexpagg",
        Path("plots") / "total_consumption_real_vs_rexpagg.png"
    )
    
    output_dir = Path(config["output_dir"])
    df = df.sort_values(by="Total consumption (annual) (real)", ascending=False)
    df.to_csv(output_dir / "consumption_aggregate.csv", index=False)

if __name__ == "__main__":
    """
    Main function to execute the script.
    """
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    main(config)