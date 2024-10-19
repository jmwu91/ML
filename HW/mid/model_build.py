# import those package we need 
import os
import numpy as np
import pandas as pd
import pickle
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm  
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance
from multiprocessing import Process, Manager
from scipy.stats import uniform, randint


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


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
# in linux
X_train = pd.read_csv("/home/jmwu/桌面/ML/HW/mid/X_train_clean.csv")
X_test = pd.read_csv("/home/jmwu/桌面/ML/HW/mid/X_test_clean.csv")
y_train = pd.read_csv("/home/jmwu/桌面/ML/HW/mid/y_train_clean.csv")
y_test = pd.read_csv("/home/jmwu/桌面/ML/HW/mid/y_test_clean.csv")







# # in windows
# X_train = pd.read_csv("X_train_clean.csv") 
# X_test = pd.read_csv("X_test_clean.csv")
# y_train = pd.read_csv("y_train_clean.csv")
# y_test = pd.read_csv("y_test_clean.csv")

# convert data type
y_train = y_train.astype(dtype = 'int')
y_test = y_test.astype(dtype = 'int')


"""
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
"""

"""def custom_score(y_true, y_pred):
    return 0.5 * f1_score(y_true, y_pred) + 0.5 * roc_auc_score(y_true, y_pred)

def grid_search_with_timeout(X_train, y_train, X_test, y_test, timeout = 1800):
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500, 700],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5],
        'scale_pos_weight': [1, 10, 50]
    }

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_jobs=12,
        random_state=20231121,
        enable_categorical=True,
        tree_method='hist'
    )

    grid_search = GridSearchCV(
        xgb_model, 
        param_grid=param_grid, 
        cv=5, 
        verbose=2,
        n_jobs=-1, 
        scoring=make_scorer(custom_score),
        return_train_score=True
    )

    def run_grid_search(grid_search, X_train, y_train, X_test, y_test, result_dict):
        grid_search.fit(
            X_train, 
            y_train, 
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            eval_metric=['auc', 'aucpr'],
            verbose=True
        )
        result_dict['best_estimator'] = grid_search.best_estimator_
        result_dict['best_params'] = grid_search.best_params_
        result_dict['best_score'] = grid_search.best_score_

    manager = Manager()
    result_dict = manager.dict()
    p = Process(target=run_grid_search, args=(grid_search, X_train, y_train, X_test, y_test, result_dict))
    p.start()
    p.join(timeout)

    if p.is_alive():
        print(f"GridSearchCV did not finish within {timeout} seconds. Terminating the process.")
        p.terminate()
        p.join()

    return result_dict

# 運行帶有超時的 GridSearchCV
start_time = time.time()
result = grid_search_with_timeout(X_train, y_train, X_test, y_test)
end_time = time.time()

print(f"Total time taken: {end_time - start_time:.2f} seconds")

# 保存結果
if 'best_estimator' in result:
    best_xgb_model_grid = result['best_estimator']
    
    # 保存模型
    with open('xgb_grid_search_model.pkl', 'wb') as f:
        pickle.dump(best_xgb_model_grid, f)

    best_xgb_model_grid.save_model('xgb_grid_search_model.json')

    # 保存最佳參數和分數
    with open('xgb_grid_search_results.json', 'w') as f:
        json.dump({
            'best_params': result['best_params'],
            'best_score': result['best_score']
        }, f, indent=4)

    # 保存特徵重要性
    feature_importance = best_xgb_model_grid.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    feature_importance_df.to_csv('xgb_grid_search_feature_importance.csv', index=False)

else:
    print("GridSearchCV was terminated before finding the best model. No results were saved.")"""


def custom_score(y_true, y_pred):
    return 0.5 * f1_score(y_true, y_pred) + 0.5 * roc_auc_score(y_true, y_pred)

class BestModelCheckpoint:
    def __init__(self, save_path):
        self.best_score = float('-inf')
        self.best_model = None
        self.save_path = save_path

    def __call__(self, estimator, _, step):
        score = estimator.best_score_
        if score > self.best_score:
            self.best_score = score
            self.best_model = estimator.best_estimator_
            self.save_model()

    def save_model(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Saved new best model with score: {self.best_score}")

def grid_search_with_timeout(X_train, y_train, X_test, y_test, timeout = 3600):
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500, 700],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5],
        'scale_pos_weight': [1, 10, 50]
    }

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_jobs=12,
        random_state=20231121,
        enable_categorical=True,
        tree_method='hist'
    )

    checkpoint = BestModelCheckpoint('best_model_checkpoint.pkl')

    grid_search = GridSearchCV(
        xgb_model, 
        param_grid=param_grid, 
        cv=5, 
        verbose=2,
        n_jobs=-1, 
        scoring=make_scorer(custom_score),
        return_train_score=True
    )

    def run_grid_search(grid_search, X_train, y_train, X_test, y_test, result_dict):
        grid_search.fit(
            X_train, 
            y_train, 
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            eval_metric=['auc', 'aucpr'],
            verbose=True,
            callbacks=[checkpoint]
        )
        result_dict['best_estimator'] = grid_search.best_estimator_
        result_dict['best_params'] = grid_search.best_params_
        result_dict['best_score'] = grid_search.best_score_

    manager = Manager()
    result_dict = manager.dict()
    p = Process(target=run_grid_search, args=(grid_search, X_train, y_train, X_test, y_test, result_dict))
    p.start()
    p.join(timeout)

    if p.is_alive():
        print(f"GridSearchCV did not finish within {timeout} seconds. Terminating the process.")
        p.terminate()
        p.join()

    return result_dict, checkpoint.best_model

def save_results(result, best_model, X_train):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = 'model_results'
    os.makedirs(save_dir, exist_ok=True)

    # 保存最佳模型（來自 checkpoint 或 GridSearchCV）
    model_to_save = best_model if best_model is not None else result.get('best_estimator')
    if model_to_save:
        model_path = os.path.join(save_dir, f'xgb_best_model_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_to_save, f)
        print(f"Best model saved at: {model_path}")

        # 保存為 JSON 格式
        json_path = os.path.join(save_dir, f'xgb_best_model_{timestamp}.json')
        model_to_save.save_model(json_path)
        print(f"Best model saved in JSON format at: {json_path}")

    # 保存最佳參數和分數
    params_score = {
        'best_params': result.get('best_params', 'Not available'),
        'best_score': result.get('best_score', 'Not available')
    }
    params_path = os.path.join(save_dir, f'xgb_best_params_{timestamp}.json')
    with open(params_path, 'w') as f:
        json.dump(params_score, f, indent=4)
    print(f"Best parameters and score saved at: {params_path}")

    # 保存特徵重要性
    if model_to_save and hasattr(model_to_save, 'feature_importances_'):
        feature_importance = model_to_save.feature_importances_
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        importance_path = os.path.join(save_dir, f'xgb_feature_importance_{timestamp}.csv')
        feature_importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved at: {importance_path}")

# 主執行部分
start_time = time.time()
result, best_model = grid_search_with_timeout(X_train, y_train, X_test, y_test)
end_time = time.time()

print(f"Total time taken: {end_time - start_time:.2f} seconds")

save_results(result, best_model, X_train)





# # XGBoost

# # RandomizedSearchCV 部分

# param_dist = {
#     'learning_rate': uniform(0.01, 0.19),
#     'n_estimators': randint(300, 1000),
#     'max_depth': randint(3, 10),
#     'min_child_weight': randint(1, 10),
#     'gamma': uniform(0, 0.5),
#     'subsample': uniform(0.6, 0.4),
#     'colsample_bytree': uniform(0.6, 0.4),
#     'reg_alpha': uniform(0, 1),
#     'reg_lambda': uniform(0, 1),
#     'scale_pos_weight': uniform(1, 5)
# }

# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     n_jobs=12,
#     seed=20231121,
#     enable_categorical=True,
#     tree_method='hist'
# )

# random_search = RandomizedSearchCV(
#     xgb_model, 
#     param_distributions=param_dist, 
#     n_iter=100, 
#     cv=5, 
#     verbose=1, 
#     n_jobs=-1, 
#     random_state=20231121,
#     scoring='f1'
# )

# random_search.fit(X_train, y_train, 
#                   eval_set=[(X_test, y_test)], 
#                   early_stopping_rounds=50,
#                   eval_metric='auc',
#                   verbose=True)

# best_xgb_model_random = random_search.best_estimator_

# # 保存 RandomizedSearchCV 結果
# with open('xgb_random_search_model.pkl', 'wb') as f:
#     pickle.dump(best_xgb_model_random, f)

# best_xgb_model_random.save_model('xgb_random_search_model.json')

# with open('xgb_random_search_results.json', 'w') as f:
#     json.dump({
#         'best_params': random_search.best_params_,
#         'best_score': random_search.best_score_
#     }, f, indent=4)

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
