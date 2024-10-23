import os
import numpy as np
import pandas as pd
import pickle
import json
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

def load_models(model_dir, timestamp):
    """載入已訓練的模型"""
    models = {}
    
    try:
        # 載入 LightGBM
        lgb_path = os.path.join(model_dir, f'lightgbm_model_{timestamp}.txt')
        models['lgb'] = lgb.Booster(model_file=lgb_path)
        
        # 載入 Random Forest
        with open(os.path.join(model_dir, f'random_forest_model_{timestamp}.pkl'), 'rb') as f:
            models['rf'] = pickle.load(f)
            
        # 載入 XGBoost
        with open(os.path.join(model_dir, f'xgb_random_search_model_{timestamp}.pkl'), 'rb') as f:
            models['xgb'] = pickle.load(f)
            
        # 載入 AdaBoost
        with open(os.path.join(model_dir, f'adaboost_model_{timestamp}.pkl'), 'rb') as f:
            models['ada'] = pickle.load(f)
            
        # 載入 Gradient Boosting
        with open(os.path.join(model_dir, f'gradient_boosting_model_{timestamp}.pkl'), 'rb') as f:
            models['gb'] = pickle.load(f)
            
        print("All models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise
    
    return models

def load_data(data_dir):
    """載入訓練和測試數據"""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_clean.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_clean.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train_clean.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test_clean.csv"))
    
    return X_train, X_test, y_train, y_test

class EnsembleOptimizer:
    """最佳化集成模型的權重和閾值"""
    def __init__(self, base_predictions, true_labels):
        self.base_predictions = base_predictions
        self.true_labels = true_labels
        self.best_weights = None
        self.best_threshold = None
        self.cv_scores = []
        
    def objective(self, trial):
        # 為每個模型分配權重
        weights = {
            'lgb': trial.suggest_float('lgb_weight', 0.1, 0.8),
            'xgb': trial.suggest_float('xgb_weight', 0.1, 0.8),
            'rf': trial.suggest_float('rf_weight', 0.1, 0.8),
            'ada': trial.suggest_float('ada_weight', 0.1, 0.8),
            'gb': trial.suggest_float('gb_weight', 0.1, 0.8)
        }
        
        # 正規化權重
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # 計算加權預測
        ensemble_pred = sum(self.base_predictions[model] * weight 
                          for model, weight in weights.items())
        
        # 優化閾值
        threshold = trial.suggest_float('threshold', 0.2, 0.8)
        predictions = (ensemble_pred >= threshold).astype(int)
        
        # 計算多個評估指標
        f1 = f1_score(self.true_labels, predictions)
        precision = precision_score(self.true_labels, predictions)
        recall = recall_score(self.true_labels, predictions)
        auc = roc_auc_score(self.true_labels, ensemble_pred)
        
        # 自定義評分函數 - 可以根據業務需求調整權重
        score = 0.4 * f1 + 0.3 * precision + 0.2 * recall + 0.1 * auc
        
        return score
    
    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, 
                      callbacks=[self._log_best_results])
        
        # 保存最佳參數
        self.best_weights = {
            'lgb': study.best_params['lgb_weight'],
            'xgb': study.best_params['xgb_weight'],
            'rf': study.best_params['rf_weight'],
            'ada': study.best_params['ada_weight'],
            'gb': study.best_params['gb_weight']
        }
        
        # 正規化權重
        total_weight = sum(self.best_weights.values())
        self.best_weights = {k: v/total_weight for k, v in self.best_weights.items()}
        self.best_threshold = study.best_params['threshold']
        
        # 繪製優化過程
        self._plot_optimization_history(study)
        
        return study.best_value, self.best_weights, self.best_threshold
    
    def _log_best_results(self, study, trial):
        """記錄優化過程中的最佳結果"""
        self.cv_scores.append(trial.value)
        if len(self.cv_scores) % 10 == 0:  # 每10次試驗打印一次
            print(f"Trial {len(self.cv_scores)}: Best score = {study.best_value:.4f}")
    
    def _plot_optimization_history(self, study):
        """繪製優化歷史"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cv_scores)
        plt.title('Optimization History')
        plt.xlabel('Trial')
        plt.ylabel('Score')
        plt.grid(True)
        plt.show()
        
        # 繪製參數重要性
        optuna.visualization.plot_param_importances(study)
        plt.show()

def get_model_predictions(models, X):
    """獲取所有基礎模型的預測"""
    predictions = {}
    for name, model in models.items():
        if name == 'lgb':
            predictions[name] = model.predict(X)
        else:
            predictions[name] = model.predict_proba(X)[:, 1]
    return predictions

def evaluate_ensemble(y_true, y_pred, y_prob):
    """評估集成模型的性能"""
    results = {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob)
    }
    
    # 混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return results

def save_ensemble_results(results, output_dir, timestamp):
    """保存集成學習的結果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整結果
    results_file = os.path.join(output_dir, f'ensemble_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        # 將numpy類型轉換為Python原生類型
        serializable_results = {
            'weights': {k: float(v) for k, v in results['weights'].items()},
            'threshold': float(results['threshold']),
            'performance': {k: float(v) for k, v in results['performance'].items()}
        }
        json.dump(serializable_results, f, indent=4)
    
    # 保存預測結果
    predictions_file = os.path.join(output_dir, f'ensemble_predictions_{timestamp}.csv')
    pd.DataFrame({
        'predicted_class': results['predictions'],
        'predicted_probability': results['probabilities']
    }).to_csv(predictions_file, index=False)
    
    print(f"Results saved to:")
    print(f"- {results_file}")
    print(f"- {predictions_file}")

def main():
    # 設定路徑
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "training_data")
    model_dir = os.path.join(current_dir, "models")
    output_dir = os.path.join(current_dir, "ensemble_results")
    
    # 設定時間戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 載入數據和模型
    print("Loading data and models...")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    models = load_models(model_dir, "20241023_030219")  # 替換為你的模型時間戳
    
    # 獲取基礎模型預測
    print("\nGetting base model predictions...")
    train_predictions = get_model_predictions(models, X_train)
    test_predictions = get_model_predictions(models, X_test)
    
    # 優化集成模型
    print("\nOptimizing ensemble...")
    optimizer = EnsembleOptimizer(train_predictions, y_train)
    best_score, weights, threshold = optimizer.optimize(n_trials=100)
    
    # 生成最終預測
    print("\nGenerating final predictions...")
    ensemble_proba = sum(test_predictions[model] * weight 
                        for model, weight in weights.items())
    ensemble_pred = (ensemble_proba >= threshold).astype(int)
    
    # 評估最終結果
    print("\nEvaluating ensemble performance...")
    performance = evaluate_ensemble(y_test, ensemble_pred, ensemble_proba)
    
    # 準備並保存結果
    results = {
        'weights': weights,
        'threshold': threshold,
        'predictions': ensemble_pred,
        'probabilities': ensemble_proba,
        'performance': performance
    }
    
    save_ensemble_results(results, output_dir, timestamp)

if __name__ == "__main__":
    main()

################################################################################################################################

# # in 3 models
# import os
# import numpy as np
# import pandas as pd
# import pickle
# import json
# import lightgbm as lgb
# import xgboost as xgb
# from datetime import datetime
# from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
# from sklearn.model_selection import KFold
# import optuna
# import matplotlib.pyplot as plt
# import seaborn as sns

# def load_models(model_dir, timestamp):
#     """載入已訓練的模型"""
#     models = {}
    
#     # 載入 LightGBM
#     lgb_path = os.path.join(model_dir, f'lightgbm_model_{timestamp}.txt')
#     models['lgb'] = lgb.Booster(model_file=lgb_path)
    
#     # 載入 Random Forest
#     with open(os.path.join(model_dir, f'random_forest_model_{timestamp}.pkl'), 'rb') as f:
#         models['rf'] = pickle.load(f)
        
#     # 載入 XGBoost
#     with open(os.path.join(model_dir, f'xgb_random_search_model_{timestamp}.pkl'), 'rb') as f:
#         models['xgb'] = pickle.load(f)
    
#     return models

# def load_data(data_dir):
#     """載入訓練和測試數據"""
#     X_train = pd.read_csv(os.path.join(data_dir, "X_train_clean.csv"))
#     X_test = pd.read_csv(os.path.join(data_dir, "X_test_clean.csv"))
#     y_train = pd.read_csv(os.path.join(data_dir, "y_train_clean.csv"))
#     y_test = pd.read_csv(os.path.join(data_dir, "y_test_clean.csv"))
    
#     return X_train, X_test, y_train, y_test

# class EnsembleOptimizer:
#     """最佳化集成模型的權重和閾值"""
#     def __init__(self, base_predictions, true_labels):
#         self.base_predictions = base_predictions
#         self.true_labels = true_labels
#         self.best_weights = None
#         self.best_threshold = None
        
#     def objective(self, trial):
#         # 為每個模型分配權重
#         weights = {
#             'lgb': trial.suggest_float('lgb_weight', 0.1, 0.8),
#             'xgb': trial.suggest_float('xgb_weight', 0.1, 0.8),
#             'rf': trial.suggest_float('rf_weight', 0.1, 0.8)
#         }
        
#         # 正規化權重
#         total_weight = sum(weights.values())
#         weights = {k: v/total_weight for k, v in weights.items()}
        
#         # 計算加權預測
#         ensemble_pred = sum(self.base_predictions[model] * weight 
#                           for model, weight in weights.items())
        
#         # 優化閾值
#         threshold = trial.suggest_float('threshold', 0.2, 0.8)
#         predictions = (ensemble_pred >= threshold).astype(int)
        
#         # 計算綜合評分
#         f1 = f1_score(self.true_labels, predictions)
#         precision = precision_score(self.true_labels, predictions)
#         recall = recall_score(self.true_labels, predictions)
#         auc = roc_auc_score(self.true_labels, ensemble_pred)
        
#         return 0.4 * f1 + 0.3 * precision + 0.2 * recall + 0.1 * auc
    
#     def optimize(self, n_trials=100):
#         study = optuna.create_study(direction='maximize')
#         study.optimize(self.objective, n_trials=n_trials)
        
#         # 保存最佳參數
#         self.best_weights = {
#             'lgb': study.best_params['lgb_weight'],
#             'xgb': study.best_params['xgb_weight'],
#             'rf': study.best_params['rf_weight']
#         }
#         total_weight = sum(self.best_weights.values())
#         self.best_weights = {k: v/total_weight for k, v in self.best_weights.items()}
#         self.best_threshold = study.best_params['threshold']
        
#         return study.best_value, self.best_weights, self.best_threshold

# def get_model_predictions(models, X):
#     """獲取所有基礎模型的預測"""
#     predictions = {}
#     for name, model in models.items():
#         if name == 'lgb':
#             predictions[name] = model.predict(X)
#         else:
#             predictions[name] = model.predict_proba(X)[:, 1]
#     return predictions

# def save_ensemble_results(results, output_dir, timestamp):
#     """保存集成學習的結果"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 保存權重和閾值
#     with open(os.path.join(output_dir, f'ensemble_params_{timestamp}.json'), 'w') as f:
#         json.dump({
#             'weights': results['weights'],
#             'threshold': results['threshold'],
#             'performance': results['performance']
#         }, f, indent=4)
    
#     # 保存預測結果
#     pd.DataFrame({
#         'predicted_class': results['predictions'],
#         'predicted_probability': results['probabilities']
#     }).to_csv(os.path.join(output_dir, f'ensemble_predictions_{timestamp}.csv'), 
#               index=False)

# def main():
#     # 設定路徑
#     current_dir = os.getcwd()
#     data_dir = os.path.join(current_dir, "training_data")
#     model_dir = os.path.join(current_dir, "models")
#     output_dir = os.path.join(current_dir, "ensemble_results")
    
#     # 設定時間戳
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # 載入數據和模型
#     print("Loading data and models...")
#     X_train, X_test, y_train, y_test = load_data(data_dir)
#     models = load_models(model_dir, "20241023_052756")  # 替換為你的模型時間戳
    
#     # 獲取基礎模型預測
#     print("Getting base model predictions...")
#     train_predictions = get_model_predictions(models, X_train)
#     test_predictions = get_model_predictions(models, X_test)
    
#     # 優化集成模型
#     print("Optimizing ensemble...")
#     optimizer = EnsembleOptimizer(train_predictions, y_train)
#     best_score, weights, threshold = optimizer.optimize()
    
#     # 生成最終預測
#     print("Generating final predictions...")
#     ensemble_proba = sum(test_predictions[model] * weight 
#                         for model, weight in weights.items())
#     ensemble_pred = (ensemble_proba >= threshold).astype(int)
    
#     # 評估結果
#     final_score = f1_score(y_test, ensemble_pred)
#     print(f"\nFinal F1 Score: {final_score:.4f}")
#     print(f"Optimal weights: {weights}")
#     print(f"Optimal threshold: {threshold:.4f}")
    
#     # 保存結果
#     results = {
#         'weights': weights,
#         'threshold': threshold,
#         'predictions': ensemble_pred,
#         'probabilities': ensemble_proba,
#         'performance': {
#             'f1_score': final_score,
#             'best_optimization_score': best_score
#         }
#     }
    
#     save_ensemble_results(results, output_dir, timestamp)
#     print(f"\nResults saved in {output_dir}")

# if __name__ == "__main__":
#     main()