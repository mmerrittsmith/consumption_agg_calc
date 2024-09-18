import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.construct_food_sub_agg import (
    clean_mrk_mod_d,
    clean_hh_mod_g1,
    calc_price,
    calculate_all_food_prices,
    standardize_units,
    strip_non_numeric,
    calc_food_consumption
)

@pytest.fixture
def sample_mrk_mod_d():
    return pd.DataFrame({
        'd1_a': ["TOMATO", "COOKING OIL", "NIKHWANI"],
        'd1_b': [408, 803, 404],
        'd1_c': ['kg', 'L', 'kg'],
        'd1_d': ["1", "9", "10B"],
        'D01': [1, 0, 1],
        'd1_b': ['101', '102', '103'],
        'd1_g': [0.5, 1.0, 0.75],
        'd1_i': [0.6, np.nan, 0.8],
        'd1_k': [0.4, 1.1, np.nan],
        'd1_h': [100, 200, 150],
        'd1_j': [110, np.nan, 160],
        'd1_l': [90, 210, np.nan]
    })

@pytest.fixture
def sample_hh_mod_g1():
    return pd.DataFrame({
        'HHID': [1, 2, 3],
        'case_id': [100, 200, 300],
        'Quantity consumed in last week': [2, 3, 2],
        'Quantity purchased': [0, 3, 1],
        'hh_g01': [1, 0, 1],
        'hh_g02': ['101', '102', '103'],
        'G04': [2, 2, 1],
        'Amount paid': [10, 12, 15]
    })

@pytest.fixture
def sample_config():
    return {
        'col_names': {
            'mrk_mod_d': {
                'MarketID': 'Unique Market ID',
                'MarketID_Visit': 'Survey Solutions Unique Visit ID',
                'D01': 'item_available',
                'd1_a': 'item_name',
                'd1_b': 'item_code',
                'd1_c': 'unit_name',
                'd1_d': 'unit_code',
                'd1_g': 'item_weight_1',
                'd1_h': 'item_price_1',
                'd1_i': 'item_weight_2',
                'd1_j': 'item_price_2',
                'd1_k': 'item_weight_3',
                'd1_l': 'item_price_3',
                'd1_m': 'item_available_market',
            },
            'hh_mod_g1': {
                'case_id': 'case_id',
                'HHID': 'HHID',
                'hh_g00_1': 'Who in the HH is most knowledgable about food consumption in the HH?',
                'hh_g00_2': 'Who in the household is reporting information on food consumption in this househ',
                'hh_g01': 'Item consumed in last week?',
                'hh_g01_oth': 'hh_g01_oth',
                'hh_g02': 'item_code',
                'hh_g03a': 'Quantity consumed in last week',
                'hh_g03b': 'unit_code',
                'hh_g03b_label': 'unit_code_label',
                'hh_g03b_oth': 'hh_g03b_oth',
                'hh_g03c': 'hh_g03c',
                'hh_g03c_1': 'hh_g03c_1',
                'hh_g04a': 'Quantity purchased',
                'hh_g04b': 'Quantity purchased in last week unit_code',
                'hh_g04b_label': 'Quantity purchased in last week unit_label',
                'hh_g04b_oth': 'hh_g04b_oth',
                'hh_g04c': 'hh_g04c',
                'hh_g04c_1': 'hh_g04c_1',
                'hh_g05': 'Amount paid',
                'hh_g06a': 'Own-production quantity',
                'hh_g06b': 'Own-production unit_code',
                'hh_g06b_label': 'Own-production unit_label',
                'hh_g06b_oth': 'hh_g06b_oth',
                'hh_g06c': 'hh_g06c',
                'hh_g06c_1': 'hh_g06c_1',
                'hh_g07a': 'Gifts and other sources quantity',
                'hh_g07b': 'Gifts and other sources unit_code',
                'hh_g07b_label': 'Gifts and other sources unit_label',
                'hh_g07b_oth': 'hh_g07b_oth',
                'hh_g07c': 'hh_g07c',
                'hh_g07c_1': 'hh_g07c_1',
            }
        }
    }

def test_clean_mrk_mod_d(sample_mrk_mod_d, sample_config):
    cleaned_df = clean_mrk_mod_d(sample_mrk_mod_d, sample_config['col_names']['mrk_mod_d'])
    assert len(cleaned_df) == 2  # Only 'Yes' items should remain
    assert 'unit_weight (kg)' in cleaned_df.columns
    assert 'unit_price' in cleaned_df.columns
    assert cleaned_df['item_code'].dtype == int

def test_clean_hh_mod_g1(sample_hh_mod_g1, sample_config):
    cleaned_df = clean_hh_mod_g1(sample_hh_mod_g1, sample_config['col_names']['hh_mod_g1'])
    assert len(cleaned_df) == 2  # Only consumed items should remain
    assert 'All purchased' in cleaned_df.columns

def test_calc_price():
    df = pd.DataFrame({
        'item_code': [1, 1, 1, 2, 2],
        'item_name': ['Apple', 'Apple', 'Apple', 'Banana', 'Banana'],
        'district': ['X', 'X', 'X', 'Y', 'Y'],
        'reside': ['Urban', 'Urban', 'Rural', 'Urban', 'Rural'],
        'region': ['A', 'A', 'B', 'A', 'B'],
        'unit_price': [10, 12, 15, 20, 25]
    })
    row = pd.Series({'item_code': 1, 'region': 'A', 'item_name': 'Apple', 'district': 'X', 'reside': 'Urban'})
    price = calc_price(df[df['item_code'] == 1], row)
    assert price == 12
    row = pd.Series({'item_code': 2, 'region': 'A', 'item_name': 'Banana', 'district': 'Y', 'reside': 'Urban'})
    price = calc_price(df[df['item_code'] == 2], row)
    assert price == 22.5

def test_calculate_all_food_prices():
    df = pd.DataFrame({
        'item_code': [1, 1, 1, 2, 2],
        'item_name': ['Apple', 'Apple', 'Apple', 'Banana', 'Banana'],
        'district': ['X', 'X', 'X', 'Y', 'Y'],
        'reside': ['Urban', 'Urban', 'Rural', 'Urban', 'Rural'],
        'region': ['A', 'A', 'B', 'A', 'B'],
        'unit_price': [10, 12, 15, 20, 25],
        "All purchased": [0, 0, 1, 0, 1],
        "Quantity consumed in last week": [2, 3, 2, 1, 4],
        "Quantity purchased": [0, 1, 2, 0, 4],
        "Amount paid": [0, 12, 30, 0, 100],
    })
    result = calculate_all_food_prices(df)
    assert result['Amount paid'].tolist() == [24, 12+24, 30, 22.5, 100]


def test_standardize_units():
    hh_mod_g1 = pd.DataFrame({
        'item_code': [1, 2],
        'unit_code': ['A', 'B'],
        'Quantity consumed in last week': [2, 3],
        'Quantity purchased': [1, 3],
    })
    mrk_mod_d = pd.DataFrame({
        'item_code': [1, 2],
        'unit_code': ['A', 'B'],
        'unit_weight (kg)': [0.5, 1.0]
    })
    result = standardize_units(hh_mod_g1, mrk_mod_d)
    assert 'Amount consumed in last week (kg)' in result.columns
    assert result['Amount consumed in last week (kg)'].tolist() == [1.0, 3.0]

def test_strip_non_numeric():
    assert strip_non_numeric('ABC123') == '123'
    assert strip_non_numeric('10.5') == '105'
    assert strip_non_numeric('') == ''


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])