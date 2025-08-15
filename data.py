from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import urllib.request
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler

from test_set_split import shuffle_and_split_data, split_data_with_id_hash
from stratify_sampling_test_train import stratify_grouping, stratify_splitting

from sklearn.linear_model import LinearRegression

# pipeline import:
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# import custom class: 
from ClusterSimilarity import ClusterSimilarity

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
plt.show()

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

# === Handling Text and Categorical Attributes: 
housing_cat = housing[['ocean_proximity']] # [[]] returns a DataFrame instead of ['ocean_prox'] which only returns a normal Series. This works because SkLearn Encoder only works with DataFrame.

# Use SkLearn OrdinalEncoder:
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

print(
    ">>> ordinal_encoder categories: \n", ordinal_encoder.categories_, "\n", # Tại sao ở dòng ordinal encoder đầu chưa gọi housing_cat mà ở đây nó đã nhận được giá trị rồi nhỉ. 
    ">>> Encoded Categories: \n", housing_cat_encoded[:8]
)

# Use SkLearn OneHotEncoder: 
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

print(housing['ocean_proximity'].info())
print(">>> Housing category 1 hot: \n", housing_cat_1hot)
print(">>> OneHotEncoder categories: \n", cat_encoder.categories_)


# ===========================================
# === Feature Scaling and Transformation ====
# ===========================================

# Break down of book's topics: 
# 1. Feature Scaling Importance.
# 2. Min-Max Scaling (Normalization)
# 3. Standardization (z-score)
# 4. Scaling Sparse Matrices
# 5. Handling Heavy Tailed distribution
# 6. Bucketizing
# 7. Adding Similarities Features (Gaussinan RBF) (Handling multi-modal distributions)
# 8. Transforming Target Values
# 9. Custom Transformers (Function Transformers)
# 10. Custom trainable transformers (BaseEstimators, TransformerMixin)
# 11. Cluster Similarity with KMeans + RBF 
# --> Understand Linear Regression, simple regression vs multiple regression. 

target_scalar = StandardScaler()
scaled_labels = target_scalar.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[['median_income']], scaled_labels)
some_new_data = housing[['median_income']].iloc[:5] # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scalar.inverse_transform(scaled_predictions)

# === Custom Transformers ====

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('standardize', StandardScaler())
])

housing_num_prepared = num_pipeline.fit_transform(housing_num)
print(housing_num_prepared[:2])

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns= num_pipeline.get_feature_names_out(),
    index=housing_num.index)

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
"total_bedrooms", "population", "households", "median_income"]

cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

def column_ratio(X): return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in): return ["ratio"]  # feature names out

def ratio_pipeline(): 
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )
    
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
    "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)), #  # type: ignore
], remainder=default_num_pipeline)


housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)
print(preprocessing.get_feature_names_out())

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
lin_reg.fit(housing, housing_labels)
housing_predictions[:5].round(-2) # type: ignore # round to the nearest hundreds. 
print(housing_labels.iloc[:5].values)