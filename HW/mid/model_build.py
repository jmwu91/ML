# import those package we need 
import os
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm  
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression


# import data
X_train = pd.read_csv("X_train_clean.csv") 
X_test = pd.read_csv("X_test_clean.csv")
y_train = pd.read_csv("y_train_clean.csv")
y_test = pd.read_csv("y_test_clean.csv")

# convert data type
y_train = y_train.astype(dtype = 'int')
y_test = y_test.astype(dtype = 'int')



# LightGBM
lgb_train = lgb.Dataset(X_train, label = y_train)
lgb_eval = lgb.Dataset(X_test, label = y_test, reference = lgb_train)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 10
}

num_round = 3000
#bst = lgb.train(params, lgb_train, num_round, valid_sets=[lgb_train, lgb_eval])
bst = lgb.train(params, lgb_train, num_round, 
                valid_sets = [lgb_train, lgb_eval], 
                callbacks = [lgb.early_stopping(stopping_rounds = 200)]
                )

# save model
bst.save_model('lightgbm_model.txt')


# # Light GBM 
# feature_importance_in_lgbm = bst.feature_importance(importance_type = 'split')  # 或 'gain'，取決於你想要的類型

# feature_names_in_lgbm = bst.feature_name()

# feature_importance_df_lgbm = pd.DataFrame({'Feature': feature_names_in_lgbm, 
#                                            'Importance': feature_importance_in_lgbm}
#                                            )

# # 依照特徵重要性降序排序
# feature_importance_df_lgbm = feature_importance_df_lgbm.sort_values(by = 'Importance', ascending = False)

# # 繪製特徵重要性圖表
# plt.figure(figsize = (10, 6))
# plt.barh(feature_importance_df_lgbm['Feature'], 
#          feature_importance_df_lgbm['Importance']
#          )

# plt.xlabel('Importance')
# plt.title('Feature Importance in lgbm')
# plt.show()

def clean_data(X_train):
    # Replace missing values with the mean of each column in: 'csmam'
    X_train = X_train.fillna({'csmam': X_train['csmam'].mean()})
    # Replace gaps forward from the previous valid value in: 'flg_3dsmk'
    X_train = X_train.fillna({'flg_3dsmk': X_train['flg_3dsmk'].ffill()})
    return X_train

X_train = clean_data(X_train.copy())

rf_classifier = RandomForestClassifier(n_estimators = 300, 
                                       random_state = 20231123, 
                                       n_jobs = 12
                                       )
rf_classifier.fit(X_train, y_train)

feature_importance_in_rf = rf_classifier.feature_importances_

# 保存模型
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

# 保存特徵重要性
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance_in_rf
})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
feature_importance_df.to_csv('rf_feature_importance.csv', index=False)

# 保存模型參數
model_params = {
    'n_estimators': rf_classifier.n_estimators,
    'random_state': rf_classifier.random_state,
    'n_jobs': rf_classifier.n_jobs
}
with open('rf_model_params.json', 'w') as f:
    json.dump(model_params, f, indent=4)

print("Random Forest model, feature importance, and model parameters have been saved.")





# XGBoost

# RandomizedSearchCV 部分

param_dist = {
    'learning_rate': uniform(0.01, 0.19),
    'n_estimators': randint(300, 1000),
    'max_depth': randint(3, 10),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
    'scale_pos_weight': uniform(1, 5)
}

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_jobs=12,
    seed=20231121,
    enable_categorical=True,
    tree_method='hist'
)

random_search = RandomizedSearchCV(
    xgb_model, 
    param_distributions=param_dist, 
    n_iter=100, 
    cv=5, 
    verbose=1, 
    n_jobs=-1, 
    random_state=20231121,
    scoring='f1'
)

random_search.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)], 
                  early_stopping_rounds=50,
                  eval_metric='auc',
                  verbose=True)

best_xgb_model_random = random_search.best_estimator_

# 保存 RandomizedSearchCV 結果
with open('xgb_random_search_model.pkl', 'wb') as f:
    pickle.dump(best_xgb_model_random, f)

best_xgb_model_random.save_model('xgb_random_search_model.json')

with open('xgb_random_search_results.json', 'w') as f:
    json.dump({
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_
    }, f, indent=4)

# # GridSearchCV 部分

# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [300, 500, 700],
#     'max_depth': [3, 5, 7],
#     'min_child_weight': [1, 3, 5],
#     'gamma': [0, 0.1, 0.2],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'reg_alpha': [0, 0.1, 0.5],
#     'reg_lambda': [0, 0.1, 0.5],
#     'scale_pos_weight': [1, 10, 50]
# }

# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     n_jobs=12,
#     random_state=20231121,
#     enable_categorical=True,
#     tree_method='hist'
# )

# def custom_score(y_true, y_pred):
#     return 0.5 * f1_score(y_true, y_pred) + 0.5 * roc_auc_score(y_true, y_pred)

# grid_search = GridSearchCV(
#     xgb_model, 
#     param_grid=param_grid, 
#     cv=5, 
#     verbose=2,
#     n_jobs=-1, 
#     scoring=make_scorer(custom_score),
#     return_train_score=True
# )

# grid_search.fit(
#     X_train, 
#     y_train, 
#     eval_set=[(X_train, y_train), (X_test, y_test)],
#     early_stopping_rounds=50,
#     eval_metric=['auc', 'aucpr'],
#     verbose=True
# )

# best_xgb_model_grid = grid_search.best_estimator_

# # 保存 GridSearchCV 結果
# with open('xgb_grid_search_model.pkl', 'wb') as f:
#     pickle.dump(best_xgb_model_grid, f)

# best_xgb_model_grid.save_model('xgb_grid_search_model.json')

# with open('xgb_grid_search_results.json', 'w') as f:
#     json.dump({
#         'best_params': grid_search.best_params_,
#         'best_score': grid_search.best_score_
#     }, f, indent=4)

# # 保存兩個模型的特徵重要性
# for name, model in [('random', best_xgb_model_random), ('grid', best_xgb_model_grid)]:
#     feature_importance = model.feature_importances_
#     feature_names = X_train.columns
#     feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
#     feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
#     feature_importance_df.to_csv(f'xgb_{name}_search_feature_importance.csv', index=False)
