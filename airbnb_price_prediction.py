import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. Data Acquisition
def load_data(url):
    try:
        df = pd.read_csv(url)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# 2. Data Exploration and Preprocessing
def preprocess_data(df):
    # Handle missing values, outliers, and data types
    # Convert price to numeric
    df['price'] = df['price'].replace(r'\$', '', regex=True).replace(',', '', regex=False).astype(float)

    # Handle missing values using median imputation
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Remove outliers (example: price > 99th percentile)
    price_threshold = df['price'].quantile(0.99)
    df = df[df['price'] <= price_threshold]

    return df

# 3. Feature Engineering
def create_features(df):
    # One-hot encoding for categorical variables
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

    # Interaction terms (example: combining location and property type)
    # df['location_property'] = df['neighbourhood_group'].astype(str) + "_" + df['room_type'].astype(str)
    # df = pd.get_dummies(df, columns=['location_property'], dummy_na=False)

    # Polynomial features (example: price squared)
    # df['price_squared'] = df['price']**2

    return df

# 4. Model Implementation
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Elastic Net Regression": ElasticNet(),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "Random Forest Regression": RandomForestRegressor(),
        "Gradient Boosting Regression": GradientBoostingRegressor()
    }
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

# 5. Hyperparameter Tuning
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# 6. Model Evaluation
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {'R-squared': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
        print(f"{name} - R-squared: {r2}, MSE: {mse}, RMSE: {rmse}, MAE: {mae}")
    return results

# 7. Analysis and Insights
def analyze_results(results):
    # Compare models and provide insights
    best_model = min(results, key=lambda k: results[k]['RMSE'])
    print(f"\nBest model: {best_model} with RMSE: {results[best_model]['RMSE']}")
    # Additional analysis can be added here, e.g., feature importance

# 8. Visualization
def visualize_data(df):
    # Create visualizations to support findings
    sns.pairplot(df)
    plt.show()
    # Additional visualizations can be added here, e.g., price distribution

# Main function
def main():
    # 1. Data Acquisition
    # Create a sample DataFrame
    data = {
        'price': [100, 200, 150, 250, 300, 120, 180, 230, 280, 320, 110, 190, 160, 240, 290, 310, 130, 170, 220, 270],
        'room_type': ['Private room', 'Entire home/apt', 'Private room', 'Entire home/apt', 'Shared room', 'Private room', 'Entire home/apt', 'Private room', 'Entire home/apt', 'Shared room', 'Private room', 'Entire home/apt', 'Private room', 'Entire home/apt', 'Shared room', 'Private room', 'Entire home/apt', 'Private room', 'Entire home/apt', 'Shared room'],
        'neighbourhood_group': ['Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Brooklyn', 'Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Brooklyn', 'Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Brooklyn', 'Manhattan', 'Brooklyn', 'Manhattan', 'Queens', 'Brooklyn'],
        'number_of_reviews': [10, 20, 5, 15, 25, 12, 18, 22, 28, 32, 11, 19, 16, 24, 29, 31, 13, 17, 21, 27],
        'latitude': [40.7, 40.6, 40.75, 40.72, 40.65, 40.71, 40.61, 40.76, 40.73, 40.66, 40.72, 40.62, 40.77, 40.74, 40.67, 40.73, 40.63, 40.78, 40.75, 40.68],
        'longitude': [-74.0, -73.9, -73.95, -73.8, -73.92, -74.01, -73.91, -73.96, -73.81, -73.93, -74.02, -73.92, -73.97, -73.82, -73.94, -74.03, -73.93, -73.98, -73.83, -73.95]
    }
    df = pd.DataFrame(data)

    # 2. Data Exploration and Preprocessing
    df = preprocess_data(df)

    # 3. Feature Engineering
    df = create_features(df)

    # Split data into training and testing sets
    X = df.drop('price', axis=1)
    y = df['price']

    # Scale numerical features
    numerical_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Model Implementation
    trained_models = train_models(X_train, y_train)

    # 5. Hyperparameter Tuning (example for Random Forest)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    tuned_rf = tune_hyperparameters(trained_models['Random Forest Regression'], param_grid, X_train, y_train)
    trained_models['Random Forest Regression'] = tuned_rf

    # 6. Model Evaluation
    results = evaluate_models(trained_models, X_test, y_test)

    # 7. Analysis and Insights
    analyze_results(results)

    # 8. Visualization
    visualize_data(df)

if __name__ == "__main__":
    main()