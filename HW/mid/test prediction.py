import pickle
import pandas as pd
import numpy as np
import json

# load models
# lightgbm










# Random forest
with open('./model_results/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

rf_feature_importance = pd.read_csv('./model_results/rf_feature_importance.csv')

with open('./model_results/rf_model_params.json', 'r') as f:
    rf_model_params = json.load(f)

print("Model loaded successfully.")
print("Top 5 important features:", rf_feature_importance.head())
print("Model parameters:", rf_model_params)
print(rf_model)

y_probabilities_rf = np.mean([estimator.predict_proba(X_test)[:, 1] for estimator in rf_classifier.estimators_], axis=0)


# xgboost


# adaboost


# gradient boost
