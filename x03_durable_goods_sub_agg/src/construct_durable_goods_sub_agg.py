import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
import yaml

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='logs/durable_goods_sub_agg.log',
                    filemode='w')
logger = logging.getLogger(__name__)

def impute_item_age(df: pd.DataFrame) -> pd.DataFrame:
    for item in df['item_code'].unique():
        mask = df['item_code'] == item
        values = df.loc[mask, "item_age"]
        if len(values) > 0:  # Only process if we have values
            df.loc[mask, "item_age"] = values.median()
        else:
            df.loc[mask, "item_age"] = 0
    return df

def calc_features(df: pd.DataFrame) -> pd.Series:
    remaining_service_years, item_lifetime = [], []
    for row_index in range(len(df)):
        row = df.iloc[row_index]
        if row["item_code"] in [517, 518]: # car and motorcycle
            item_lifetime.append(row["item_age"]*3)
            remaining_service_years.append(row["item_age"]*3-row["item_age"])
        else: # everything else
            item_lifetime.append(row["item_age"]*2)
            remaining_service_years.append(row["item_age"]*2-row["item_age"])
        if remaining_service_years[-1] < 0:
            remaining_service_years[-1] = 2
    df["item_lifetime"] = item_lifetime
    df["remaining_service_years"] = remaining_service_years
    df["annual_use_value"] = df["money_that_could_be_made_by_selling_item"] / df["remaining_service_years"]
    return df

def winsorize_by_item(df: pd.DataFrame, col: str, limits=(0.05, 0.05)) -> pd.DataFrame:
    """
    Winsorize a column by group, preserving the distribution within each item_code.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        col (str): Name of column to winsorize
        limits (tuple): Lower and upper percentile limits for winsorization
        
    Returns:
        pd.DataFrame: DataFrame with winsorized values
    """
    for item in df['item_code'].unique():
        mask = df['item_code'] == item
        values = df.loc[mask, col]
        if len(values) > 0:  # Only process if we have values
            lower = np.percentile(values, limits[0] * 100)
            upper = np.percentile(values, (1 - limits[1]) * 100)
            df.loc[mask, col] = np.clip(values, lower, upper)
    return df

def main(config: Dict[str, Any]) -> None:
    data_dir = Path.cwd().parent / config["data_dir"]
    hh_mod_a = pd.read_stata(data_dir / "hh_mod_a_filt.dta", convert_categoricals=True)
    durable_goods = pd.read_stata(data_dir / "HH_MOD_L.dta")
    durable_goods = durable_goods.rename(columns=config["cols"])
    durable_goods = durable_goods[durable_goods["purchased_item"] == "Yes"]
    df = durable_goods.merge(hh_mod_a, how="left", on=["case_id", "HHID"])
    # remaining service years = estimated lifetime of good- current age
    # If this is negative, set it to 2 years
    # For motorcycles and cars, assume the lifetime is 3x current age
    df = impute_item_age(df)
    df = winsorize_by_item(df, "money_that_could_be_made_by_selling_item")
    df = winsorize_by_item(df, "item_count")
    df = calc_features(df)
    hh_level_df = df[["HHID", "annual_use_value"]].groupby("HHID").sum()
    consumption_agg_df = pd.read_stata(data_dir / "ihs5_consumption_aggregate.dta", convert_categoricals=False)
    consumption_agg_df = consumption_agg_df[["HHID", "price_indexL"]]
    hh_level_df = hh_level_df.merge(consumption_agg_df, how="left", on="HHID")
    hh_level_df["Durable Goods Consumption (annual) (nominal)"] = hh_level_df["annual_use_value"]
    hh_level_df["Durable Goods Consumption (annual) (real)"] = hh_level_df["Durable Goods Consumption (annual) (nominal)"]/hh_level_df["price_indexL"]
    hh_level_df = hh_level_df[["HHID", "Durable Goods Consumption (annual) (nominal)", "Durable Goods Consumption (annual) (real)"]]
    hh_level_df.to_csv("outputs/durable_goods_sub_agg.csv")
    df.to_csv("outputs/durable_goods_level_df.csv")

if __name__ == "__main__":
    """
    Main function to execute the script.
    """
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    main(config)