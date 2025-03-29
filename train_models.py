import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the dataset
df = pd.read_csv('comprehensive_mutual_funds_data.csv')

# Data preprocessing
# Replace empty strings with NaN
df = df.replace('', np.nan)

# Convert numeric columns to float
numeric_cols = ['min_sip', 'min_lumpsum', 'expense_ratio', 'fund_size_cr', 
                'fund_age_yr', 'sortino', 'alpha', 'sd', 'beta', 'sharpe', 
                'risk_level', 'rating', 'returns_1yr', 'returns_3yr', 'returns_5yr']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Feature engineering
# Create categorical features
df['amc_size'] = df.groupby('amc_name')['fund_size_cr'].transform('sum')
df['amc_fund_count'] = df.groupby('amc_name')['scheme_name'].transform('count')

# One-hot encode categorical features
categorical_cols = ['category', 'sub_category']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and targets
feature_cols = [col for col in df_encoded.columns if col not in 
                ['scheme_name', 'fund_manager', 'amc_name', 'returns_1yr', 'returns_3yr', 'returns_5yr']]

# Function to train and save a model
def train_and_save_model(X, y, model_name, test_size=0.2, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create a pipeline with imputer, scaler, and model
    if model_name == '1yr':
        model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    elif model_name == '3yr':
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    else:  # 5yr
        model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name} Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("-" * 50)
    
    # Save the model
    with open(f'models/model_{model_name}.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline

# Train and save models for different time horizons
print("Training 1-year returns prediction model...")
X_1yr = df_encoded.dropna(subset=['returns_1yr'])[feature_cols]
y_1yr = df_encoded.dropna(subset=['returns_1yr'])['returns_1yr']
model_1yr = train_and_save_model(X_1yr, y_1yr, '1yr')

print("Training 3-year returns prediction model...")
X_3yr = df_encoded.dropna(subset=['returns_3yr'])[feature_cols]
y_3yr = df_encoded.dropna(subset=['returns_3yr'])['returns_3yr']
model_3yr = train_and_save_model(X_3yr, y_3yr, '3yr')

print("Training 5-year returns prediction model...")
X_5yr = df_encoded.dropna(subset=['returns_5yr'])[feature_cols]
y_5yr = df_encoded.dropna(subset=['returns_5yr'])['returns_5yr']
model_5yr = train_and_save_model(X_5yr, y_5yr, '5yr')

# Save feature columns for later use
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("All models trained and saved successfully!") 