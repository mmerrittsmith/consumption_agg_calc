import pandas as pd
from pathlib import Path
import yaml
import numpy as np
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Add logging
# TODO: Add unit tests
# TODO: Add type hints where they are missing
# TODO: Review output to ensure it is correct
# TODO: I don't see an aggregate consumption column to compare this to

def main(config: dict):
    """
    Main function to process housing data and calculate rental values.
    
    This function:
    1. Loads and merges household and housing survey data
    2. Processes rental data for both renters and non-renters
    3. Creates and applies a rent prediction model
    4. Identifies and handles outliers in rental values
    5. Saves the processed data and prediction model
    
    Args:
        config (dict): Configuration dictionary containing data paths and column mappings
        
    Outputs:
        - Saves processed household data to 'output/hh_level_df.csv'
        - Saves rent prediction model to 'output/rent_predictor_model.pkl'
    """
    data_dir = Path.cwd().parent / config["data_dir"]
    hh_mod_a = pd.read_stata(data_dir / "hh_mod_a_filt.dta", convert_categoricals=True)
    housing = pd.read_stata(data_dir / "HH_MOD_F.dta")
    housing = housing.rename(columns=config["cols"])
    df = housing.merge(hh_mod_a, how="left", on=["case_id", "HHID"])
    df['interviewDate'] = pd.to_datetime(df['interviewDate'], errors='coerce')
    df['survey_year'] = df['interviewDate'].dt.year
    df['survey_month'] = df['interviewDate'].dt.month
    renter_df = df[df["property_use_type"] == "RENTED"]
    renter_df["rent_per_month"] = get_rent_per_month(renter_df, True)
    # It's not clear to me that I should winsorize this column, as it's possible that
    # the highest values are correct, but they seem unlikely.
    # renter_df = winsorize_column(renter_df, "rent_per_month", limits=(0.025, 0.025))
    sns.displot(renter_df, x="rent_per_month")
    plt.savefig(Path("plots") / "rent_per_month_for_renters_distribution.png")
    plt.close()

    model, renter_df["predicted_rent"] = make_rent_predictor(renter_df, "rent_per_month", config)
    non_renter_df = df[df["property_use_type"] != "RENTED"]
    non_renter_df["rent_per_month"] = get_rent_per_month(non_renter_df, False)
    non_renter_df = non_renter_df.sort_values(by="rent_per_month", ascending=False)
    non_renter_df.to_csv(Path("output") / "non_renter_df.csv")
    non_renter_df["predicted_rent"] = predict_rent(non_renter_df, model, config)
    df = pd.concat([renter_df, non_renter_df])
    df['residuals'] = df['rent_per_month'] - df['predicted_rent']
    df['outliers'] = np.abs(df['residuals']) > 2 * df['residuals'].std()
    df[df["outliers"]].to_csv(Path("output") / "rent_outliers.csv")
    df["rent_coalesced"] = [pred_rent if outlier else rent for pred_rent, rent, outlier in zip(df["predicted_rent"], df["rent_per_month"], df["outliers"])]
    consumption_agg_df = pd.read_stata(data_dir / "ihs5_consumption_aggregate.dta", convert_categoricals=False)
    consumption_agg_df = consumption_agg_df[["HHID", "price_indexL"]]
    df = df.merge(consumption_agg_df, how="left", on="HHID")
    df["Housing Consumption (annual) (nominal)"] = df["rent_coalesced"]*12
    df["Housing Consumption (annual) (real)"] = df["Housing Consumption (annual) (nominal)"]/df["price_indexL"]
    create_rent_analysis_plot(df)

    df = df[["HHID", "case_id", "Housing Consumption (annual) (nominal)", "Housing Consumption (annual) (real)"]]
    df.to_csv('output/housing_sub_agg.csv')
    with open('output/rent_predictor_model.pkl', 'wb') as file:
        pickle.dump(model, file)

def create_rent_analysis_plot(df: pd.DataFrame, output_path: str = 'plots/rent_analysis.png') -> None:
    """
    Creates a scatter plot comparing predicted vs reported rent values.
    Only includes households that reported rent (renters).
    
    Args:
        df: DataFrame containing rent data
        output_path: Path to save the plot
    """
    # Filter for only renters (those who reported rent)
    renters_df = df[df["property_use_type"] == "RENTED"].copy()
    
    plt.figure(figsize=(10, 6))
    
    # Calculate correlations
    pearson_corr = renters_df["predicted_rent"].corr(renters_df["rent_per_month"], method='pearson')
    spearman_corr = renters_df["predicted_rent"].corr(renters_df["rent_per_month"], method='spearman')
    
    # Create scatter plot
    sns.scatterplot(data=renters_df, 
                   x="predicted_rent",
                   y="rent_per_month",
                   alpha=0.5)
    
    # Add diagonal line for perfect prediction
    max_val = max(renters_df["predicted_rent"].max(), renters_df["rent_per_month"].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect prediction (x=y)')
    
    # Add title and labels
    plt.title(f"Predicted vs Reported Monthly Rent\nPearson correlation: {pearson_corr:.4f}\nSpearman correlation: {spearman_corr:.4f}")
    plt.xlabel("Predicted monthly rent")
    plt.ylabel("Reported monthly rent")
    
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()


#     sns.histplot(df, x="rent_coalesced")
#     plt.savefig(Path("plots") / "rent_coalesced_distribution.png")
#     plt.close()

def get_rent_per_month(df: pd.DataFrame, renter: bool):
    """
    Converts rent payments from different time units to monthly amounts.
    
    Args:
        df (pd.DataFrame): DataFrame containing rent payment data
        renter (bool): If True, uses columns for actual renters. If False, uses columns 
                      for hypothetical rent that owners could receive.
    
    Returns:
        list: List of monthly rent amounts converted from various time units (day, week, 
              month, year) to monthly values. Uses conversion factors:
              - Day to month: multiply by 30.44 (average days in month)
              - Week to month: multiply by 4
              - Month: no conversion needed
              - Year to month: divide by 12
    """
    rent_per_month = []
    if renter:
        rent_col = "amount_paid_to_rent_property"
        time_unit_col = "rent_time_unit_2"
    else:
        rent_col = "amount_owner_could_receive_in_rent"
        time_unit_col = "rent_time_unit"
    for i in df.index.tolist():
        if df.loc[i, time_unit_col] == "DAY":
            rent_per_month.append(df.loc[i, rent_col]*30.44)
        elif df.loc[i, time_unit_col] == "WEEK":
            rent_per_month.append(df.loc[i, rent_col]*4)
        elif df.loc[i, time_unit_col] == "MONTH":
            rent_per_month.append(df.loc[i, rent_col])
        elif df.loc[i, time_unit_col] == "YEAR":
            rent_per_month.append(df.loc[i, rent_col] / 12)
    return rent_per_month

def prep_df_for_rent_prediction(df: pd.DataFrame, independent_vars: list[str]):
    """
    Prepares features and target variables for rent prediction modeling by converting categorical variables to dummy variables.

    Args:
        df (pd.DataFrame): Input DataFrame containing rent and predictor variables
        independent_vars (list[str]): List of column names to use as predictor variables

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): Feature matrix with dummy variables for categorical predictors
            - y (pd.Series): Target variable (log of rent amount)
    """
    X = df[independent_vars]
    non_numeric_cols = [col for col in independent_vars if not pd.api.types.is_numeric_dtype(X[col])]
    X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True, dtype=float)
    try: 
        y = df["log_rent"]
        return X, y
    except KeyError:
        return X

def make_rent_predictor(df, target_col, config):
    """
    Creates a regression model to predict rental values based on housing characteristics.
    
    The model takes the log of actual rent for renters and regresses it on various 
    housing characteristics and fixed effects:
    
    Housing characteristics:
    - Type of dwelling features:
        - Roof type
        - Wall material 
        - Floor type
        - Number of rooms
    - Source of drinking water
    - Type of toilet
    - Electricity availability (in town and dwelling)
    
    Fixed effects:
    - Urban/rural indicator
    - Region
    - District 
    - Survey year
    - Survey month
    
    This model is used to predict outliers in self-reported rental data.

    Args:
        df (pd.DataFrame): DataFrame containing housing survey data
        target_col (str): Name of column containing actual rent values
        config (dict): Configuration dictionary containing column mappings

    Returns:
        tuple: A tuple containing:
            - model (sm.OLS): Fitted OLS regression model
            - predicted_rent (np.array): Array of predicted rent values
    """
    df['log_rent'] = np.log(df[target_col])

    independent_vars = [
        config['cols']['hh_f08'],  # type_of_roof
        config['cols']['hh_f07'],  # outer_walls_material
        config['cols']['hh_f09'],  # type_of_floor
        config['cols']['hh_f10'],  # num_rooms_occupied
        config['cols']['hh_f11'],  # lighting_fuel_source (proxy for electricity availability)
        config['cols']['hh_f27'],  # does_village_have_electricity
        config['cols']['hh_f36_1'],  # drinking_water_source
        config['cols']['hh_f41'],  # type_of_toilet
        "region",
        "district",
        "reside",
        "survey_year",
        "survey_month"
    ]
    X, y = prep_df_for_rent_prediction(df, independent_vars)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    predicted_log_rent = model.predict(X)
    predicted_rent = np.exp(predicted_log_rent)
    return model, predicted_rent

def predict_rent(df, model, config):
    """
    Predicts rental values for housing units using a pre-fitted regression model.
    
    Uses housing characteristics and location data to predict rental values. The model
    takes into account physical attributes of the dwelling (roof, walls, floor, rooms),
    utilities (electricity, water, toilet), and geographic factors (region, district,
    urban/rural status).

    Args:
        df (pd.DataFrame): DataFrame containing housing survey data
        model (sm.OLS): Pre-fitted OLS regression model for rent prediction
        config (dict): Configuration dictionary containing column mappings

    Returns:
        np.array: Array of predicted rental values in original scale (not log-transformed)
    """
    independent_vars = [
        config['cols']['hh_f08'],  # type_of_roof
        config['cols']['hh_f07'],  # outer_walls_material
        config['cols']['hh_f09'],  # type_of_floor
        config['cols']['hh_f10'],  # num_rooms_occupied
        config['cols']['hh_f11'],  # lighting_fuel_source (proxy for electricity availability)
        config['cols']['hh_f27'],  # does_village_have_electricity
        config['cols']['hh_f36_1'],  # drinking_water_source
        config['cols']['hh_f41'],  # type_of_toilet
        "region",
        "district",
        "reside",
        "survey_year",
        "survey_month"
    ]
    X = prep_df_for_rent_prediction(df, independent_vars)
    X = sm.add_constant(X)

    predicted_log_rent = model.predict(X)
    predicted_rent = np.exp(predicted_log_rent)
    # sns.displot(predicted_rent)
    # plt.savefig(Path("plots") / "predicted_rent_distribution.png")
    # plt.close()
    X['predicted_rent'] = predicted_rent
    X = X.sort_values(by="predicted_rent", ascending=False)
    X.to_csv(Path("output") / "predicted_rent_df.csv")
    predicted_rent = winsorize_column(pd.DataFrame({"predicted_rent": predicted_rent}), "predicted_rent", limits=(0.01, 0.01))["predicted_rent"]
    X['predicted_rent'] = predicted_rent
    X.to_csv(Path("output") / "predicted_rent_df_winsorized.csv")
    print(predicted_rent.describe())
    return predicted_rent

def winsorize_column(df: pd.DataFrame, col: str, limits=(0.05, 0.05)) -> pd.DataFrame:
    """
    Winsorize a column by group, preserving the distribution within each item_code.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        col (str): Name of column to winsorize
        limits (tuple): Lower and upper percentile limits for winsorization
        
    Returns:
        pd.DataFrame: DataFrame with winsorized values
    """
    print(df[col].describe())
    values = df[col]
    if len(values) > 0: 
        lower = np.percentile(values, limits[0] * 100)
        upper = np.percentile(values, (1 - limits[1]) * 100)
        df[col] = np.clip(values, lower, upper)
    print(df[col].describe())
    return df

if __name__ == "__main__":
    """
    Main function to execute the script.
    """
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    main(config)