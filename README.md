# Mutual Fund Returns Predictor

A machine learning-powered web application that predicts 1-year, 3-year, and 5-year returns for mutual funds based on their characteristics. This project uses Flask for the backend, scikit-learn for machine learning models, and Bootstrap for the frontend.

## Features

- Predict future returns for mutual funds across three time horizons (1-year, 3-year, 5-year)
- Interactive and user-friendly web interface
- Pre-filled form with recommended values for easy use
- Visualize predictions with interactive charts
- Responsive design for all device sizes
- API endpoint for programmatic access

## Implementation Details

### Machine Learning Models

The application uses three separate models for different time horizons:

1. **1-Year Returns Model**: Gradient Boosting Regressor
2. **3-Year Returns Model**: Random Forest Regressor
3. **5-Year Returns Model**: Gradient Boosting Regressor

The models are trained on a comprehensive dataset of mutual funds, considering various features:
- Fund characteristics (expense ratio, fund size, fund age)
- Risk metrics (sortino ratio, alpha, standard deviation, beta, sharpe ratio)
- Categorical information (AMC, category, sub-category)

### Project Structure 