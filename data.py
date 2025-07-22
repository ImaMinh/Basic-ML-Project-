from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
from test_set_split import shuffle_and_split_data, split_data_with_id_hash

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
plt.show()

# =======================================================
# === Divide the Dataset into Train-set and Test-set ====
# =======================================================

train_set, test_set = shuffle_and_split_data(housing, 0.2)

# Using the split_data_with_id (crc32) function to split the dataset. 
# However, the original dataset doesn't have an 'index' column that we can access and 
# manipulate. Thus, we need to re-structure the dataset. 

housing_with_id = housing.reset_index(drop=False)
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, 'index')
