import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Create a dummy dataframe
data = {'age': [22, 38, 26, 35, 35],
        'fare': [7.25, 71.28, 7.92, 53.10, 8.05],
        'pclass': [3, 1, 3, 1, 3],
        'sex': [0, 1, 1, 0, 0], # 0 for male, 1 for female
        'sibsp': [1, 1, 0, 1, 0],
        'parch': [0, 0, 0, 0, 0],
        'survived': [0, 1, 1, 1, 0]}
df = pd.DataFrame(data)

# Display the first few rows of the dataframe
print(df.head())

# Print summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())