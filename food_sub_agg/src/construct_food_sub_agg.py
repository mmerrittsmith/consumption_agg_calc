import pandas as pd
import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
import yaml

#NOTE: The stripping of non-numeric characters when mergin the market data to the food consumption data seems to 
#      flatten things like red and white onions to one itemcode, or different types of rice to one item code. 
#      I'm not sure if that's a significant issue or not.

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='logs/construct_consumption_agg.log',
                    filemode='w')
logger = logging.getLogger(__name__)

def clean_mrk_mod_d(mrk_mod_d: pd.DataFrame, col_names: dict) -> pd.DataFrame:
    """
    Clean and preprocess the market data (mrk_mod_d DataFrame).

    This function performs the following operations:
    1. Filters rows where 'D01' is 'Yes'.
    2. Renames columns for clarity and consistency.
    3. Calculates median weights and prices across multiple columns.
    4. Groups and aggregates data by item and unit.
    5. Cleans and converts the 'item_code' to integer type.

    Args:
        mrk_mod_d (pd.DataFrame): Raw market data DataFrame.

    Returns:
        pd.DataFrame: Cleaned and preprocessed market data.
    """
    mrk_mod_d = mrk_mod_d.rename(columns=col_names)
    mrk_mod_d = mrk_mod_d[mrk_mod_d["item_available"] == 1]
    mrk_mod_d["item_name"] = mrk_mod_d["item_name"].str.strip()
    mrk_mod_d["median_weight (kg)"] = mrk_mod_d[["item_weight_1", "item_weight_2", "item_weight_3"]].median(axis=1, skipna=True)
    mrk_mod_d["median_price"] = mrk_mod_d[["item_price_1", "item_price_2", "item_price_3"]].median(axis=1, skipna=True)
    mrk_mod_d = mrk_mod_d[["item_code", "item_name", "unit_code", "unit_name", "median_weight (kg)", "median_price"]].groupby(["item_name", "item_code", "unit_name", "unit_code"]).median().reset_index()
    mrk_mod_d = mrk_mod_d.rename(columns={"median_weight (kg)": "unit_weight (kg)", "median_price": "unit_price"})
    mrk_mod_d['item_code'] = mrk_mod_d['item_code'].apply(strip_non_numeric)
    mrk_mod_d['item_code'] = mrk_mod_d['item_code'].astype(int)
    mrk_mod_d['unit_price'] = mrk_mod_d['unit_price'].astype(float)
    return mrk_mod_d

def clean_hh_mod_g1(hh_mod_g1: pd.DataFrame, col_names: dict) -> pd.DataFrame:
    """
    Clean and preprocess the household module G1 data.

    This function performs the following operations:
    1. Renames columns according to the provided dictionary.
    2. Filters rows to only those where 'Item consumed in last week?' is 1.
    3. Creates a new column 'All purchased' indicating if all consumed quantity was purchased.
    4. Removes rows with missing or 'nan' item codes.

    Args:
        hh_mod_g1 (pd.DataFrame): Raw household module G1 data.
        col_names (dict): Dictionary mapping old column names to new column names.

    Returns:
        pd.DataFrame: Cleaned and preprocessed household module G1 data.
    """
    hh_mod_g1 = hh_mod_g1.rename(columns=col_names)
    hh_mod_g1 = hh_mod_g1[hh_mod_g1["Item consumed in last week?"] == 1]
    hh_mod_g1["All purchased"] = hh_mod_g1["Quantity purchased"] == hh_mod_g1["Quantity consumed in last week"]
    hh_mod_g1 = hh_mod_g1[hh_mod_g1["item_code"].notna()]
    hh_mod_g1 = hh_mod_g1[hh_mod_g1["item_code"] != "nan"]
    hh_mod_g1["Amount paid"] = hh_mod_g1["Amount paid"].astype(float)
    return hh_mod_g1



def calc_price(df: pd.DataFrame, row: pd.Series) -> float:
    """
    Calculate the price for a given item based on available data.

    This function determines the price of an item by looking at increasingly broader
    geographical areas until it finds enough observations to make a reliable estimate.
    It starts with the most specific geographical area and broadens its search if
    insufficient data is found.

    Args:
        df (pd.DataFrame): The dataset containing price information for various items.
        row (pd.Series): A single row from the main dataset, containing information
                         about the specific item and its geographical location.

    Returns:
        float: The calculated price for the item. Returns np.nan if no price can be determined.

    Note:
        The function uses a geographical specificity hierarchy: region > district > reside/stratum > country.
        It requires at least 30 observations to calculate a price at any geographical level.
        If no price can be determined even at the country level, it returns np.nan.
    """
    num_obs_available = 0
    geographical_specificity_index = 0
    geographical_specificity_dict = {1: "region", 2: "district", 3: "reside", 4: "country"}
    price = np.nan
    
    while num_obs_available < 30:
        geographical_specificity_index += 1
        if geographical_specificity_index == 4:
            logger.warning(f"No prices available for {row['item_name']} for region {row['region']}, district {row['district']}, or strata {row['reside']}")
            num_obs_available = len(df)
            try:
                price = np.nanmedian(df["unit_price"])
                logger.info(f"Using median price from whole country instead, got {price}")
            except ValueError:
                logger.error("No prices available, returning nan")
            return price
        num_obs_available = len(df[df[geographical_specificity_dict[geographical_specificity_index]] == row[geographical_specificity_dict[geographical_specificity_index]]])
        logger.debug(f"Number of observations available: {num_obs_available}")
    price = np.nanmedian(df[df[geographical_specificity_dict[geographical_specificity_index]] == row[geographical_specificity_dict[geographical_specificity_index]]]["unit_price"])
    logger.info(f"Price computed normally for {row['item_name']} at {geographical_specificity_dict[geographical_specificity_index]} level, got {price}")
    return price



def calculate_all_food_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate prices for all food items in the DataFrame.

    This function processes a DataFrame containing food consumption data and calculates
    prices for items that were not fully purchased. It uses the `calc_price` function
    to estimate prices based on geographical specificity when necessary.

    Args:
        df (pd.DataFrame): A DataFrame containing food consumption data.
            Expected to have columns: 'All purchased', 'Amount paid', 'Quantity consumed in last week',
            'Quantity purchased', 'item_code', 'region', 'district', 'reside', and 'item_name'.

    Returns:
        pd.DataFrame: The input DataFrame with updated 'Amount paid' values for items
            that were not fully purchased.

    Note:
        This function is computationally intensive and may take several minutes to run
        on large datasets. Consider caching results or vectorizing if performance becomes an issue.
    """
    items_where_not_all_purchased = df[df["All purchased"] == False]
    full_amount_paid = items_where_not_all_purchased["Amount paid"].values
    full_amount_paid = full_amount_paid.astype(float)
    for i in range(len(items_where_not_all_purchased.index)):
        row = items_where_not_all_purchased.iloc[i]
        quantity_not_purchased_but_consumed = row["Quantity consumed in last week"] - row["Quantity purchased"]
        calcd_price = calc_price(df[(df["item_code"] == row["item_code"]) & (df["unit_code"] == row["unit_code"])], row)
        full_amount_paid[i] += float(quantity_not_purchased_but_consumed*calcd_price)
    items_where_not_all_purchased["Amount paid"] = full_amount_paid
    df = pd.concat([df[df["All purchased"] == True], items_where_not_all_purchased])
    df = df.sort_index()
    return df

def standardize_units(hh_mod_g1: pd.DataFrame, mrk_mod_d: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the units of consumption for household food items.

    This function takes two DataFrames: one containing household consumption data (hh_mod_g1)
    and another containing market data on unit weights (mrk_mod_d). It merges these
    DataFrames and calculates the amount of food consumed in kilograms.

    Args:
        hh_mod_g1 (pd.DataFrame): DataFrame containing household consumption data.
            Expected to have columns: 'item_code', 'unit_code', 'Quantity consumed in last week'.
        mrk_mod_d (pd.DataFrame): DataFrame containing market data with unit weights.
            Expected to have columns: 'item_code', 'unit_code', 'unit_weight (kg)'.

    Returns:
        pd.DataFrame: The input hh_mod_g1 DataFrame merged to mrk_mod_d with an additional column
            'Amount consumed in last week (kg)' representing the standardized
            consumption in kilograms.

    Note:
        This function assumes that the units to kilograms ratio is homogeneous across the nation and time.
        It performs a left merge between hh_mod_g1 and mrk_mod_d based on 'item_code' and 'unit_code'.
    """
    hh_mod_g1 = hh_mod_g1.merge(mrk_mod_d, how="left", on=["item_code", "unit_code"])
    hh_mod_g1["Amount consumed in last week (kg)"] = hh_mod_g1["Quantity consumed in last week"]*hh_mod_g1["unit_weight (kg)"]
    return hh_mod_g1

def strip_non_numeric(value: str) -> str:
    """
    Remove all non-numeric characters from a string value.

    This function takes a string input and removes all characters that are not
    digits (0-9). It's useful for cleaning data that may contain mixed
    alphanumeric values but only the numeric part is needed.

    Args:
        value (str): The input string to be processed.

    Returns:
        str: A string containing only the numeric characters from the input.

    Example:
        >>> strip_non_numeric("ABC123")
        "123"
        >>> strip_non_numeric("10.5")
        "105"
    """
    return re.sub(r'\D', '', str(value))

def calc_food_consumption(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate food consumption data at both food item and household levels.

    This function processes data from multiple Stata files to compute food consumption
    statistics. It performs the following steps:
    1. Reads and cleans household and market data.
    2. Merges datasets and standardizes units.
    3. Calculates food prices and consumption amounts.
    4. Aggregates data at both food item and household levels.

    We aren't doing outlier detection for right now

    Args:
        config (dict): Configuration dictionary containing paths to data files and other parameters.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - food_level_df: DataFrame with food item level consumption data.
            - hh_level_df: DataFrame with household level consumption data,
              including annual food consumption estimates.
    """
    data_dir = Path.cwd().parent / "MWI_2019_IHS-V_v06_M_Stata"
    hh_mod_a = pd.read_stata(data_dir / "hh_mod_a_filt.dta", convert_categoricals=False)
    hh_mod_g1 = pd.read_stata(data_dir / "HH_MOD_G1.dta", convert_categoricals=False)
    hh_mod_g1 = clean_hh_mod_g1(hh_mod_g1, config["col_names"]["hh_mod_g1"])
    df = hh_mod_g1.merge(hh_mod_a, how="left", on=["case_id", "HHID"])
    mrk_mod_d = clean_mrk_mod_d(pd.read_stata(data_dir / "mrk_mod_d.dta", convert_categoricals=False), config["col_names"]["mrk_mod_d"])
    df = standardize_units(df, mrk_mod_d)
    food_level_df = calculate_all_food_prices(df)
    hh_level_df = df[["HHID", "case_id", "Amount paid", "Amount consumed in last week (kg)"]].groupby("HHID").sum()
    hh_level_df["Food consumption (annual)"] = hh_level_df["Amount paid"]*52
    return food_level_df, hh_level_df


def analyze_food_consumption(food_level_df: pd.DataFrame, hh_level_df: pd.DataFrame) -> None:
    """
    Analyze the correlation between food consumption and rexp_cat01.

    This function calculates the correlation between 'Food consumption (annual)' and 'rexp_cat01'
    and plots a scatter plot with a line of best fit. It also prints summary statistics and the
    ratio of means between the two variables.

    Args:
        food_level_df (pd.DataFrame): DataFrame with food item level consumption data.
        hh_level_df (pd.DataFrame): DataFrame with household level consumption data.
    """
    data_dir = Path.cwd().parent / "MWI_2019_IHS-V_v06_M_Stata"
    consumption_agg_df = pd.read_stata(data_dir / "ihs5_consumption_aggregate.dta", convert_categoricals=False)
    hh_level_df = hh_level_df.merge(consumption_agg_df, how="left", on="HHID")
    correlation = hh_level_df['Food consumption (annual)'].corr(hh_level_df['rexp_cat01'])
    print(f"The correlation between 'Food consumption (annual)' and 'rexp_cat01' is: {correlation:.4f}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Food consumption (annual)', y='rexp_cat01', data=hh_level_df, alpha=0.5)

    # Add a line of best fit
    x = hh_level_df['Food consumption (annual)']
    y = hh_level_df['rexp_cat01']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8)

    plt.title(f"Food consumption (annual) vs rexp_cat01\nCorrelation: {correlation:.4f}")
    plt.xlabel("Food consumption (annual) (Our estimate)")
    plt.ylabel("rexp_cat01 (IHS-V)")

    plt.tight_layout()
    plt.savefig("plots/food_consumption_vs_rexp_cat01.png")

    print("\nSummary Statistics:")
    print(hh_level_df[['Food consumption (annual)', 'rexp_cat01']].describe())

    mean_ratio = hh_level_df['Food consumption (annual)'].mean() / hh_level_df['rexp_cat01'].mean()
    print(f"\nRatio of means (Food consumption (annual) / rexp_cat01): {mean_ratio:.4f}")

def write_food_consumption_agg(food_level_df: pd.DataFrame, hh_level_df: pd.DataFrame) -> None:
    food_level_df.to_csv("outputs/food_level_df.csv", index=False)
    hh_level_df.to_csv("outputs/hh_level_df.csv", index=False)

if __name__ == "__main__":
    """
    Main function to execute the script.
    """
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    food_level_df, hh_level_df = calc_food_consumption(config)
    analyze_food_consumption(food_level_df, hh_level_df)
    write_food_consumption_agg(food_level_df, hh_level_df)