import pandas as pd
import numpy as np
from pathlib import Path
import yaml

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
    return merged_df

def main(config: dict) -> None:
    df = read_files(config)
    df["Total consumption (annual) (nominal)"] = df[[x for x in df.columns if "Consumption (annual) (nominal)" in x]].sum(axis=1)
    df["Total consumption (annual) (real)"] = df[[x for x in df.columns if "Consumption (annual) (real)" in x]].sum(axis=1)
    output_dir = Path(config["output_dir"])
    df.to_csv(output_dir / "consumption_aggregate.csv", index=False)

if __name__ == "__main__":
    """
    Main function to execute the script.
    """
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    main(config)