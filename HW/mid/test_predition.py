import os
import numpy as np
import pandas as pd
import pickle
import json
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_saved_models(model_dir, timestamp):
    """載入已儲存的模型"""
    print("開始載入模型...")
    models = {}
    
    try:
        model_paths = {
            'lgb': f'lightgbm_model_{timestamp}.txt',
            'rf': f'random_forest_model_{timestamp}.pkl',
            'xgb': f'xgb_random_search_model_{timestamp}.pkl',
            'ada': f'adaboost_model_{timestamp}.pkl',
            'gb': f'gradient_boosting_model_{timestamp}.pkl'
        }
        
        for model_name, path in model_paths.items():
            full_path = os.path.join(model_dir, path)
            if os.path.exists(full_path):
                if model_name == 'lgb':
                    models[model_name] = lgb.Booster(model_file=full_path)
                else:
                    with open(full_path, 'rb') as f:
                        models[model_name] = pickle.load(f)
                print(f"成功載入 {model_name.upper()} 模型")
            else:
                print(f"警告: {model_name.upper()} 模型文件不存在 ({full_path})")
        
        if not models:
            raise ValueError("沒有找到任何可用的模型文件")
        
        return models
    
    except Exception as e:
        print(f"載入模型時發生錯誤: {str(e)}")
        raise

class FraudPredictor:
    def __init__(self, models, threshold=0.5, weights=None):
        """
        初始化詐欺預測器
        
        Args:
            models (dict): 已訓練的模型字典
            threshold (float): 分類閾值
            weights (dict): 各模型的權重
        """
        self.models = models
        self.threshold = threshold
        self.weights = weights or self._get_default_weights(len(models))
        self.prediction_cache = {}
        
    def _get_default_weights(self, n_models):
        """獲取默認的模型權重"""
        weight = 1.0 / n_models
        return {model: weight for model in self.models.keys()}
    
    # predict 方法的修正版本
    def predict(self, X):
        """進行集成預測"""
        predictions = {}
        
        try:
            # LightGBM prediction
            if 'lgb' in self.models:
                predictions['lgb'] = self.models['lgb'].predict(X)
            
            # Random Forest prediction
            if 'rf' in self.models:
                predictions['rf'] = self.models['rf'].predict_proba(X)[:, 1]
            
            # XGBoost prediction
            if 'xgb' in self.models:
                predictions['xgb'] = self.models['xgb'].predict_proba(X)[:, 1]
            
            # AdaBoost prediction
            if 'ada' in self.models:
                predictions['ada'] = self.models['ada'].predict_proba(X)[:, 1]
            
            # Gradient Boosting prediction
            if 'gb' in self.models:
                predictions['gb'] = self.models['gb'].predict_proba(X)[:, 1]
            
            self.prediction_cache = predictions
            
            # 計算加權預測結果
            weighted_pred = sum(
                predictions[model] * weight 
                for model, weight in self.weights.items()
                if model in predictions
            )
            
            return weighted_pred
            
        except Exception as e:
            print(f"預測過程發生錯誤: {str(e)}")
            raise
    
    def visualize_predictions(self, probabilities):
        """視覺化各模型的預測分布"""
        plt.figure(figsize=(12, 6))
        for model_name, probs in probabilities.items():
            sns.kdeplot(probs, label=model_name.upper())
        plt.title('預測概率分布')
        plt.xlabel('預測概率')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def create_submission(self, X, output_path, include_probabilities=True):
        """創建提交文件"""
        try:
            # 獲取預測概率
            probabilities = self.predict(X)
            predictions = (probabilities >= self.threshold).astype(int)
            
            # 創建基本提交DataFrame
            submission = pd.DataFrame({
                'index': range(len(predictions)),
                'label': predictions
            })
            
            # # 如果需要，添加預測概率
            # if include_probabilities:
            #     submission['probability'] = probabilities
                
            #     # 添加各模型的單獨預測
            #     individual_predictions = self.prediction_cache
            #     for model_name, pred in individual_predictions.items():
            #         submission[f'{model_name}_prob'] = pred
            
            # 保存預測結果
            submission.to_csv(output_path, index=False, encoding='utf-8')
            print(f"預測結果已保存至: {output_path}")
            
            # 預測統計
            print("\n預測統計:")
            print(f"總樣本數: {len(predictions):,}")
            print(f"預測為詐欺交易數: {predictions.sum():,}")
            print(f"預測為正常交易數: {len(predictions) - predictions.sum():,}")
            print(f"詐欺交易比例: {predictions.mean():.4%}")
            
            # 視覺化預測分布
            self.visualize_predictions(self.prediction_cache)
            
            # 保存預測配置
            config_path = output_path.replace('.csv', '_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'threshold': self.threshold,
                    'weights': self.weights,
                    'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model_count': len(self.models),
                    'stats': {
                        'total_samples': int(len(predictions)),
                        'fraud_predictions': int(predictions.sum()),
                        'normal_predictions': int(len(predictions) - predictions.sum()),
                        'fraud_ratio': float(predictions.mean())
                    }
                }, f, indent=4)
            
            return submission
            
        except Exception as e:
            print(f"創建提交文件時發生錯誤: {str(e)}")
            raise

def main():
    try:
        # 設定路徑
        current_dir = os.getcwd()
        model_dir = os.path.join(current_dir, "models")
        result_dir = os.path.join(current_dir, "result")
        os.makedirs(result_dir, exist_ok=True)
        
        # 設定模型時間戳
        model_timestamp = "20241023_052756"
        
        # 載入測試數據
        print("載入測試數據...")
        X_test_pub = pd.read_csv("X_test_pub.csv")
        
        # 載入模型
        models = load_saved_models(model_dir, model_timestamp)
        
        # 設定模型權重
        weights = {
            'lgb': 0.25,
            'rf': 0.20,
            'xgb': 0.20,
            'ada': 0.15,
            'gb': 0.20
        }
        
        # 初始化預測器
        predictor = FraudPredictor(
            models=models,
            threshold=0.5,
            weights=weights
        )
        
        # 生成預測時間戳
        pred_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建提交文件
        output_path = os.path.join(result_dir, f'ensemble_prediction_{pred_timestamp}.csv')
        submission = predictor.create_submission(X_test_pub, output_path)
        
        print("\n預測完成！")
        print("\n預測結果預覽:")
        print(submission.head())
        
        # 視覺化預測分布
        predictor.visualize_predictions(predictor.prediction_cache)
        
    except Exception as e:
        print(f"執行過程發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()

################################################################################################

# # in 3 models
# import os
# import numpy as np
# import pandas as pd
# import pickle
# import json
# import lightgbm as lgb
# import xgboost as xgb
# from datetime import datetime
# from sklearn.ensemble import RandomForestClassifier

# def load_saved_models(model_dir, timestamp):
#     """載入已儲存的模型"""
#     print("開始載入模型...")
#     models = {}
    
#     try:
#         # 載入 LightGBM 模型
#         lgb_path = os.path.join(model_dir, f'lightgbm_model_{timestamp}.txt')
#         if os.path.exists(lgb_path):
#             models['lgb'] = lgb.Booster(model_file=lgb_path)
#             print("成功載入 LightGBM 模型")
        
#         # 載入 Random Forest 模型
#         rf_path = os.path.join(model_dir, f'random_forest_model_{timestamp}.pkl')
#         if os.path.exists(rf_path):
#             with open(rf_path, 'rb') as f:
#                 models['rf'] = pickle.load(f)
#             print("成功載入 Random Forest 模型")
        
#         # 載入 XGBoost 模型
#         xgb_path = os.path.join(model_dir, f'xgb_random_search_model_{timestamp}.pkl')
#         if os.path.exists(xgb_path):
#             with open(xgb_path, 'rb') as f:
#                 models['xgb'] = pickle.load(f)
#             print("成功載入 XGBoost 模型")
        
#         return models
    
#     except Exception as e:
#         print(f"載入模型時發生錯誤: {str(e)}")
#         raise

# class FraudPredictor:
#     def __init__(self, models, threshold=0.5, weights=None):
#         self.models = models
#         self.threshold = threshold
#         self.weights = weights or self._get_default_weights(len(models))
        
#     def _get_default_weights(self, n_models):
#         weight = 1.0 / n_models
#         return {model: weight for model in self.models.keys()}
    
#     def predict(self, X):
#         predictions = {}
        
#         try:
#             # LightGBM prediction
#             if 'lgb' in self.models:
#                 predictions['lgb'] = self.models['lgb'].predict(X)
            
#             # Random Forest prediction
#             if 'rf' in self.models:
#                 predictions['rf'] = np.mean([
#                     estimator.predict_proba(X)[:, 1] 
#                     for estimator in self.models['rf'].estimators_
#                 ], axis=0)
            
#             # XGBoost prediction
#             if 'xgb' in self.models:
#                 predictions['xgb'] = self.models['xgb'].predict_proba(X)[:, 1]
            
#             weighted_pred = sum(
#                 predictions[model] * weight 
#                 for model, weight in self.weights.items()
#                 if model in predictions
#             )
            
#             return weighted_pred
            
#         except Exception as e:
#             print(f"預測過程發生錯誤: {str(e)}")
#             raise
    
#     def create_submission(self, X, output_path):
#         try:
#             # 獲取預測概率
#             probabilities = self.predict(X)
#             predictions = (probabilities >= self.threshold).astype(int)
            
#             # 創建提交DataFrame
#             submission = pd.DataFrame({
#                 'index': range(len(predictions)),
#                 'label': predictions
#             })
            
#             # 保存預測結果
#             submission.to_csv(output_path, index=False, encoding='utf-8')
#             print(f"預測結果已保存至: {output_path}")
            
#             # 打印預測統計
#             print("\n預測統計:")
#             print(f"總樣本數: {len(predictions)}")
#             print(f"預測為詐欺交易數: {predictions.sum()}")
#             print(f"預測為正常交易數: {len(predictions) - predictions.sum()}")
#             print(f"詐欺交易比例: {predictions.mean():.4%}")
            
#             return submission
            
#         except Exception as e:
#             print(f"創建提交文件時發生錯誤: {str(e)}")
#             raise

# def main():
#     try:
#         # 設定路徑
#         current_dir = os.getcwd()
#         model_dir = os.path.join(current_dir, "models")
#         result_dir = os.path.join(current_dir, "result")
#         os.makedirs(result_dir, exist_ok=True)
        
#         # 設定模型時間戳（替換為你的時間戳）
#         model_timestamp = "20241023_052756"  # 例如: "20240323_143022"
        
#         # 載入測試數據
#         print("載入測試數據...")
#         X_test_pub = pd.read_csv("X_test_pub.csv")
        
#         # 載入模型
#         models = load_saved_models(model_dir, model_timestamp)
        
#         # 設定模型權重
#         if 'xgb' in models:
#             weights = {'lgb': 0.34, 'rf': 0.33, 'xgb': 0.33}
#         else:
#             weights = {'lgb': 0.5, 'rf': 0.5}
        
#         # 初始化預測器
#         predictor = FraudPredictor(
#             models=models,
#             threshold=0.5,  # 可以根據需要調整閾值
#             weights=weights
#         )
        
#         # 生成預測時間戳
#         pred_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # 創建提交文件
#         output_path = os.path.join(result_dir, f'ensemble_prediction_{pred_timestamp}.csv')
#         submission = predictor.create_submission(X_test_pub, output_path)
        
#         print("\n預測完成！")
#         print("\n預測結果預覽:")
#         print(submission.head())
        
#     except Exception as e:
#         print(f"執行過程發生錯誤: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()