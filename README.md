# Disney Box Office Forecast – DataDash 2026

Description:
------------
Regression-based forecasting model to predict inflation-
adjusted gross revenue for a hypothetical Disney film 
(e.g., 2032 release).

Developed for Project Greenlight – DataDash 2026.


What This Code Does:
--------------------
- Cleans and preprocesses historical Disney movie data
- Encodes categorical features:
    • Genre (multi-genre supported)
    • MPAA Rating
    • Season
- Applies log-transformed Ridge Regression
- Prevents negative revenue predictions

Model Validation:
-----------------
- 5-fold Cross Validation
- 80/20 Train-Test Split
- Hyperparameter tuning using RidgeCV

Outputs:
--------
- Predicted revenue for specified future release year
- Visualization:
    X-axis → Release Year
    Y-axis → Inflation-Adjusted Gross
- Model performance metrics (CV R², Test R²)


Model Approach:
---------------
Target Variable: Inflation Adjusted Gross
Transformation : log1p()
Inverse        : expm1()
Model          : Ridge Regression (L2 regularization)
Encoding       : One-Hot Encoding


Tech Stack:
-----------
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

