from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the trained models
with open('models/model_1yr.pkl', 'rb') as f:
    model_1yr = pickle.load(f)

with open('models/model_3yr.pkl', 'rb') as f:
    model_3yr = pickle.load(f)

with open('models/model_5yr.pkl', 'rb') as f:
    model_5yr = pickle.load(f)

with open('models/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# Load the dataset for reference values and categories
df = pd.read_csv('comprehensive_mutual_funds_data.csv')
categories = sorted(df['category'].unique())
sub_categories = sorted(df['sub_category'].unique())
amc_names = sorted(df['amc_name'].unique())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', 
                              categories=categories,
                              sub_categories=sub_categories,
                              amc_names=amc_names)
    else:
        # Get form data
        try:
            min_sip = float(request.form.get('min_sip', 0))
            min_lumpsum = float(request.form.get('min_lumpsum', 0))
            expense_ratio = float(request.form.get('expense_ratio', 0))
            fund_size_cr = float(request.form.get('fund_size_cr', 0))
            fund_age_yr = float(request.form.get('fund_age_yr', 0))
            sortino = float(request.form.get('sortino', 0))
            alpha = float(request.form.get('alpha', 0))
            sd = float(request.form.get('sd', 0))
            beta = float(request.form.get('beta', 0))
            sharpe = float(request.form.get('sharpe', 0))
            risk_level = int(request.form.get('risk_level', 1))
            rating = int(request.form.get('rating', 0))
            category = request.form.get('category')
            sub_category = request.form.get('sub_category')
            amc_name = request.form.get('amc_name')
            
            # Create a dataframe with the input data
            input_data = {
                'min_sip': min_sip,
                'min_lumpsum': min_lumpsum,
                'expense_ratio': expense_ratio,
                'fund_size_cr': fund_size_cr,
                'fund_age_yr': fund_age_yr,
                'sortino': sortino,
                'alpha': alpha,
                'sd': sd,
                'beta': beta,
                'sharpe': sharpe,
                'risk_level': risk_level,
                'rating': rating,
                'category': category,
                'sub_category': sub_category,
                'amc_name': amc_name
            }
            
            # Create a DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Feature engineering (same as in training)
            # Estimate amc_size and amc_fund_count from the original dataset
            amc_stats = df.groupby('amc_name').agg({
                'fund_size_cr': 'sum',
                'scheme_name': 'count'
            }).reset_index()
            amc_stats.columns = ['amc_name', 'amc_size', 'amc_fund_count']
            
            # Merge with input data
            input_df = pd.merge(input_df, amc_stats, on='amc_name', how='left')
            
            # One-hot encode categorical features
            input_encoded = pd.get_dummies(input_df, columns=['category', 'sub_category'], drop_first=True)
            
            # Ensure all feature columns from training are present
            for col in feature_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Select only the features used in training
            input_features = input_encoded[feature_cols]
            
            # Make predictions
            pred_1yr = model_1yr.predict(input_features)[0]
            pred_3yr = model_3yr.predict(input_features)[0]
            pred_5yr = model_5yr.predict(input_features)[0]
            
            # Round predictions to 2 decimal places
            pred_1yr = round(pred_1yr, 2)
            pred_3yr = round(pred_3yr, 2)
            pred_5yr = round(pred_5yr, 2)
            
            # Prepare result data
            result = {
                'pred_1yr': pred_1yr,
                'pred_3yr': pred_3yr,
                'pred_5yr': pred_5yr,
                'input_data': input_data
            }
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            return render_template('predict.html', 
                                  error=f"Error in prediction: {str(e)}",
                                  categories=categories,
                                  sub_categories=sub_categories,
                                  amc_names=amc_names)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        # Create a dataframe with the input data
        input_df = pd.DataFrame([data])
        
        # Feature engineering (same as in training)
        # Estimate amc_size and amc_fund_count from the original dataset
        amc_stats = df.groupby('amc_name').agg({
            'fund_size_cr': 'sum',
            'scheme_name': 'count'
        }).reset_index()
        amc_stats.columns = ['amc_name', 'amc_size', 'amc_fund_count']
        
        # Merge with input data
        input_df = pd.merge(input_df, amc_stats, on='amc_name', how='left')
        
        # One-hot encode categorical features
        input_encoded = pd.get_dummies(input_df, columns=['category', 'sub_category'], drop_first=True)
        
        # Ensure all feature columns from training are present
        for col in feature_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Select only the features used in training
        input_features = input_encoded[feature_cols]
        
        # Make predictions
        pred_1yr = float(model_1yr.predict(input_features)[0])
        pred_3yr = float(model_3yr.predict(input_features)[0])
        pred_5yr = float(model_5yr.predict(input_features)[0])
        
        return jsonify({
            'predictions': {
                '1_year': round(pred_1yr, 2),
                '3_year': round(pred_3yr, 2),
                '5_year': round(pred_5yr, 2)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 