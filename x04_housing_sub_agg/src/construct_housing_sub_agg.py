import pandas as pd
from pathlib import Path
import yaml
import numpy as np
import statsmodels.api as sm
import pickle

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
    model, renter_df["predicted_rent"] = make_rent_predictor(renter_df, "amount_paid_to_rent_property", config)
    non_renter_df = df[df["property_use_type"] != "RENTED"]
    non_renter_df["rent_per_month"] = get_rent_per_month(non_renter_df, False)
    non_renter_df["predicted_rent"] = predict_rent(non_renter_df, model, config)
    df = pd.concat([renter_df, non_renter_df])
    df['residuals'] = df['rent_per_month'] - df['predicted_rent']
    df['outliers'] = np.abs(df['residuals']) > 2 * df['residuals'].std()
    df["rent_coalesced"] = [pred_rent if outlier else rent for pred_rent, rent, outlier in zip(df["predicted_rent"], df["rent_per_month"], df["outliers"])]
    consumption_agg_df = pd.read_stata(data_dir / "ihs5_consumption_aggregate.dta", convert_categoricals=False)
    consumption_agg_df = consumption_agg_df[["HHID", "price_indexL"]]
    df = df.merge(consumption_agg_df, how="left", on="HHID")
    df["Housing Consumption (annual) (nominal)"] = df["rent_coalesced"]*12
    df["Housing Consumption (annual) (real)"] = df["Housing Consumption (annual) (nominal)"]/df["price_indexL"]
    df = df[["HHID", "case_id", "Housing Consumption (annual) (nominal)", "Housing Consumption (annual) (real)"]]
    df.to_csv('output/housing_sub_agg.csv')
    with open('output/rent_predictor_model.pkl', 'wb') as file:
        pickle.dump(model, file)

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
    y = df["log_rent"]
    non_numeric_cols = [col for col in independent_vars if not pd.api.types.is_numeric_dtype(X[col])]
    X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True, dtype=float)
    return X, y

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
    df['log_rent'] = np.log(df["rent_per_month"])
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

    predicted_log_rent = model.predict(X)
    predicted_rent = np.exp(predicted_log_rent)
    return predicted_rent

if __name__ == "__main__":
    """
    Main function to execute the script.
    """
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    main(config)