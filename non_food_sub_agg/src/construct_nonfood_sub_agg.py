import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any
import yaml

# health services, drugs, housing utilities (gas, firewood, water, electricity), transport (fuel, maintenance, repairs) (operation costs not purchases of durable items), clothing and footwear
# public transportation, communication services, recreation and cultural services (except durables), hotel and lodging, misc goods and services like soap or umbrellas.
# If the good has a reference period less than a year, annualize it. 

# Don't include sporadic expenditures like weddings, funerals, and births. Don't include remittances. Don't include expenditure to repair or updgrade dwellings.

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='logs/nonfood_sub_agg.log',
                    filemode='w')
logger = logging.getLogger(__name__)

def process_nonfood_df(filename: str, config: Dict[str, Any]) -> pd.DataFrame:
    filepath = Path.cwd().parent / config["data_dir"] / filename
    df = pd.read_stata(filepath, convert_categoricals=True)
    df = df.rename(columns=config["df_configs"][filepath.stem]["columns"])
    df = df[df["item_purchased"] == "Yes"]
    df["amount_paid_per_day"] = df["amount_paid"]/config['df_configs'][filepath.stem]["recall_period"]
    df_pivoted = df.pivot_table(
        index='case_id',
        columns='item_code',
        values='amount_paid_per_day',
        aggfunc='sum',
        fill_value=0,
        observed=True
    ).add_prefix('nfa_amount_paid_per_day_')
    df_pivoted = df_pivoted.reset_index()
    return df_pivoted

def make_full_nonfood_df(nonfood_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame: 
    for df_name in config["df_configs"]:
        df_filepath = df_name + ".dta"
        nonfood_df = nonfood_df.merge(process_nonfood_df(df_filepath, config), how="left", on="case_id")
    nonfood_df["Nonfood Consumption Total (annual)"] = (nonfood_df[[x for x in nonfood_df.columns if x.startswith("nfa_amount_paid_per_day")]].sum(axis=1))*365
    for area in config["consumption_areas"]:
        nonfood_df[f"{area} Consumption (annual)"] = (nonfood_df[[f"nfa_amount_paid_per_day_{x}" for x in config["consumption_areas"][area]["cols"]]]).sum(axis=1)*365
    nonfood_df = nonfood_df.drop(columns=["nfa_amount_paid_per_day_"+x for x in ["Repairs & maintenance to dwelling", 
                                                                                 "Repairs to household and personal items (radios, watches, etc, excluding battery purchases)",
                                                                                 "Mortgage - regular payment to purchase house",
                                                                                 "Losses to theft (value of items or cash lost)",
                                                                                 "Fines or legal fees",
                                                                                 "Lobola (bridewealth) costs",
                                                                                 "Marriage ceremony costs",
                                                                                 "Funeral costs, household members",
                                                                                 "Funeral costs, nonhousehold members (relatives, neighbors/friends)"]])
    nonfood_df.to_csv("outputs/nonfood_df.csv")
    return nonfood_df

def analyze_nonfood_consumption(nonfood_df: pd.DataFrame):
    # Note that it's fine that a few of these correlations are nan, because the true nonfood consumption for those areas is 0 too.
    data_dir = Path.cwd().parent / "MWI_2019_IHS-V_v06_M_Stata"
    consumption_agg_df = pd.read_stata(data_dir / "ihs5_consumption_aggregate.dta", convert_categoricals=False)
    nonfood_df = nonfood_df.merge(consumption_agg_df, how="left", on="HHID")
    for area in config["consumption_areas"]:
        print(nonfood_df[config["consumption_areas"][area]["code"]].describe())
        correlation = nonfood_df[f"{area} Consumption (annual)"].corr(nonfood_df[config["consumption_areas"][area]["code"]])
        print(f"{area} correlation: {correlation}")

def main(config: Dict[str, Any]) -> None:
    data_dir = Path.cwd().parent / config["data_dir"]
    hh_mod_a = pd.read_stata(data_dir / "hh_mod_a_filt.dta", convert_categoricals=True)
    nonfood_df = make_full_nonfood_df(hh_mod_a, config)
    nonfood_df.to_csv("outputs/nonfood_df.csv")
    analyze_nonfood_consumption(nonfood_df)

if __name__ == "__main__":
    """
    Main function to execute the script.
    """
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    main(config)