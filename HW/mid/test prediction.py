import pickle
import pandas as pd
import json

# 加載模型
with open('random_forest_model.pkl', 'rb') as f:
    loaded_rf_model = pickle.load(f)

# 加載特徵重要性
feature_importance = pd.read_csv('rf_feature_importance.csv')

# 加載模型參數
with open('rf_model_params.json', 'r') as f:
    model_params = json.load(f)

print("Model loaded successfully.")
print("Top 5 important features:", feature_importance.head())
print("Model parameters:", model_params)