from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import urllib.request
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from test_set_split import shuffle_and_split_data, split_data_with_id_hash
from stratify_sampling_test_train import stratify_grouping, stratify_splitting


# ======================================== DATA PREPROCESSING ==========================================



# ==========================
# === Load housing Data ====
# ==========================

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# ========================
# === Inspecting Data ====
# ========================

print(">>> Inspecting head: \n", housing.head(), "\n")
print(">>> Housing info: \n", housing.info(), "\n") # Useful to get a quick description of the data, particularly the total number of rows, each attribute's type, ...
print(">>> Describe: \n", housing.describe(), "\n")

# ocean_proximity is special because it type is object --> it can hold any kind of Python obj --> Good idea to inspect it: 
print(housing['ocean_proximity'].value_counts())

# ============================
# === Plotting Histograms ====
# ============================

housing.hist(bins=50, grid=True)
#plt.show()

# =======================================================
# === Divide the Dataset into Train-set and Test-set ====  # ---- IMPORTANT PART -----
# =======================================================

# === Random Sampling ===

# Normal splitting using ratio -> Read our test_split.txt file for analysis. 

train_set, test_set = shuffle_and_split_data(housing, 0.2)

# Using the split_data_with_id (crc32) function to split the dataset. 
# However, the original dataset doesn't have an 'index' column that we can access and 
# manipulate. Thus, we need to re-structure the dataset. 

housing_with_id = housing.reset_index(drop=False)
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, 'index')

# === Stratify Sampling ===

housing = stratify_grouping(housing)

# -- plotting histogram of this housing income --

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
#plt.show()

# --> Sau khi nhóm các nhóm thu nhập thành các stratas rồi, chúng ta sẽ bắt đầu stratify
# sampling để được test/train split đều nhau. 

strat_train_set, strat_test_set = stratify_splitting(housing)[0]

print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))


# ===========================================================================
# === Plotting Scatter Plot of Housing Distribution in the Given Dataset ====
# ===========================================================================

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
    s=housing["population"] / 100, label="population",
    c="median_house_value", cmap="jet", colorbar=True,
    legend=True, sharex=False, figsize=(10, 7))

#plt.show()


# =============================
# === Correlation Analysis ====
# =============================

# === Calculating Correlation using Pearson's r ===

corr_matrix = housing.loc[:, housing.columns != 'ocean_proximity'].corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# === Checking correlation using (Pandas's) scatter_matrix function ===

attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
#plt.show()

# --- Looking at the median_house_value row, we see that only the correlation between this and median_income has a positive corr -> plot it ---
housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1, grid=True)


# =========================================================
# === Prepare the Data for Machine Learning Algorithms ====
# =========================================================

housing = strat_train_set.drop("median_house_value", axis=1) # dropping the (target column) - labels. <predictors/features> 
housing_labels = strat_train_set["median_house_value"].copy() # getting the target column to predict. <label/target> 

# === Using SimpleImputer to fill missing values with Median (read analysis in Notes to understand more) ====
imputer = SimpleImputer(strategy='median')

# Because the median can only be computed on numerical values, we need to extract data with only the numerical attributes. 
housing_num = housing.select_dtypes(include=[np.number])

# Eventhough we know that only total_bedrooms had missing values, we cannot be sure that there won't be any missing values in the new data if the data is changed, so it's safer to apply the imputer to all numerical attributes. 
imputer.fit(housing_num) # imputer uses fit method to compute the median of each attribute and stored the result in the <statistics_> instance var. 

print(">>> imputer statistics: \n", imputer.statistics_)
print(">>> housing median statistics: \n", housing_num.median())

# After using fit() to calculate the median of each df columns, we use transform to 'really' apply the median values to the columns: 
imputer.transform(housing_num)