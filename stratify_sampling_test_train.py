import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# === Grouping the Data into Groups for stratify sampling ===

def stratify_grouping(data: pd.DataFrame) -> pd.DataFrame: 
    housing = data
    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins =[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], # thử bỏ np.infinity (np.inf) đi xem sao
                                   labels=[1, 2, 3, 4, 5])
    
    housing['income_cat'].value_counts().sort_index()
    
    return housing 

# === Stratify Selectioning ===

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []

def stratify_splitting(housing: pd.DataFrame):
    for train_index, test_index in splitter.split(housing, housing['income_cat']):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append([strat_train_set_n, strat_test_set_n])

    return strat_splits

