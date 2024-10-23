import os
import numpy as np
import pandas as pd
import pickle
import json
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

class CreditCardFraudPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.models = {}
        self.predictions = {}
        self.best_thresholds = {}
        self.common_threshold = None
        
    def load_models(self, timestamp):
        """Load all trained models"""
        # Load LightGBM
        lgb_path = os.path.join(self.model_dir, f'lightgbm_model_{timestamp}.txt')
        self.models['lgb'] = lgb.Booster(model_file=lgb_path)
        
        # Load Random Forest
        with open(os.path.join(self.model_dir, f'random_forest_model_{timestamp}.pkl'), 'rb') as f:
            self.models['rf'] = pickle.load(f)
            
        # Load XGBoost
        with open(os.path.join(self.model_dir, f'xgb_random_search_model_{timestamp}.pkl'), 'rb') as f:
            self.models['xgb'] = pickle.load(f)
            
        # Load AdaBoost
        with open(os.path.join(self.model_dir, f'adaboost_model_{timestamp}.pkl'), 'rb') as f:
            self.models['ada'] = pickle.load(f)
            
        # Load Gradient Boosting
        with open(os.path.join(self.model_dir, f'gradient_boosting_model_{timestamp}.pkl'), 'rb') as f:
            self.models['gb'] = pickle.load(f)
            
        print("All models loaded successfully")
    
    def predict_probabilities(self, X):
        """Get probability predictions from all models"""
        self.predictions['lgb'] = self.models['lgb'].predict(X)
        self.predictions['rf'] = np.mean([estimator.predict_proba(X)[:, 1] 
                                        for estimator in self.models['rf'].estimators_], axis=0)
        self.predictions['xgb'] = self.models['xgb'].predict_proba(X)[:, 1]
        self.predictions['ada'] = self.models['ada'].predict_proba(X)[:, 1]
        self.predictions['gb'] = self.models['gb'].predict_proba(X)[:, 1]
        return self.predictions
    
    def find_best_threshold(self, y_true, y_prob, thresholds):
        """Find optimal threshold for binary classification"""
        best_f1 = 0
        best_threshold = 0
        best_precision = 0
        best_recall = 0
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision_score(y_true, y_pred)
                best_recall = recall_score(y_true, y_pred)
                
        return {
            'threshold': best_threshold,
            'f1_score': best_f1,
            'precision': best_precision,
            'recall': best_recall
        }
    
    def optimize_thresholds(self, X, y, threshold_range=np.arange(0.01, 1, 0.01)):
        """Find optimal thresholds for all models"""
        predictions = self.predict_probabilities(X)
        
        # Find best threshold for each model
        print("\nOptimal thresholds for individual models:")
        print("-" * 50)
        for model_name, y_prob in predictions.items():
            metrics = self.find_best_threshold(y, y_prob, threshold_range)
            self.best_thresholds[model_name] = metrics
            print(f"{model_name.upper()}:")
            print(f"Threshold: {metrics['threshold']:.3f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print("-" * 50)
        
        # Plot F1 scores vs thresholds
        self.plot_threshold_optimization(y, threshold_range)
        
        # Find common threshold
        common_range = np.arange(0.3, 0.7, 0.01)
        f1_scores_common = []
        
        for threshold in common_range:
            f1_scores = []
            for y_prob in predictions.values():
                y_pred = (y_prob >= threshold).astype(int)
                f1_scores.append(f1_score(y, y_pred))
            f1_scores_common.append(np.mean(f1_scores))
        
        self.common_threshold = common_range[np.argmax(f1_scores_common)]
        print(f"\nBest common threshold: {self.common_threshold:.3f}")
        
        # Print results with common threshold
        self.print_metrics_with_threshold(y, self.common_threshold)
        
        # Plot correlation matrix of predictions
        self.plot_prediction_correlation()
    
    def plot_threshold_optimization(self, y, thresholds):
        """Plot F1 scores vs thresholds for all models"""
        plt.figure(figsize=(12, 6))
        colors = {
            'lgb': 'blue',
            'xgb': 'orange',
            'rf': 'green',
            'ada': 'red',
            'gb': 'purple'
        }
        
        for model_name, y_prob in self.predictions.items():
            f1_scores = [f1_score(y, (y_prob >= t).astype(int)) for t in thresholds]
            plt.plot(thresholds, f1_scores, label=f"{model_name.upper()}", color=colors[model_name])
            
            # Add vertical lines for best thresholds
            best_threshold = self.best_thresholds[model_name]['threshold']
            plt.axvline(x=best_threshold, color=colors[model_name], 
                       linestyle='--', alpha=0.5)
        
        plt.title('F1 Score vs Threshold for All Models')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_prediction_correlation(self):
        """Plot correlation matrix of model predictions"""
        pred_df = pd.DataFrame(self.predictions)
        correlation_matrix = pred_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Model Predictions')
        plt.show()
    
    def print_metrics_with_threshold(self, y, threshold):
        """Print precision, recall, and F1 score using specified threshold"""
        print("\nMetrics with common threshold:")
        print("-" * 50)
        for model_name, y_prob in self.predictions.items():
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_prob)
            
            print(f"{model_name.upper()}:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"AUC-ROC: {auc:.4f}")
            print("-" * 50)
    
    def predict(self, X, use_common_threshold=True):
        """Make final predictions using ensemble of models"""
        predictions = self.predict_probabilities(X)
        
        # Updated weights based on model performance
        weights = {
            'lgb': 0.25,
            'rf': 0.2,
            'xgb': 0.2,
            'ada': 0.15,
            'gb': 0.2
        }
        
        ensemble_pred = sum(predictions[model] * weight 
                          for model, weight in weights.items())
        
        threshold = self.common_threshold if use_common_threshold else 0.5
        return (ensemble_pred >= threshold).astype(int), ensemble_pred
    
    def save_predictions(self, predictions, probabilities, output_path):
        """Save predictions to CSV"""
        pd.DataFrame({
            'predicted_class': predictions,
            'predicted_probability': probabilities
        }).to_csv(output_path, index=False)
        
    def save_model_metrics(self, timestamp):
        """Save model metrics and thresholds to JSON"""
        metrics_path = os.path.join(self.model_dir, f'model_metrics_{timestamp}.json')
        
        metrics = {
            'best_thresholds': self.best_thresholds,
            'common_threshold': float(self.common_threshold),
            'timestamp': timestamp
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nModel metrics saved to: {metrics_path}")

def main():
    # Set paths
    current_dir = os.getcwd()
    model_dir = os.path.join(current_dir, "models")
    data_dir = os.path.join(current_dir, "training_data")
    
    # Load test data
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_clean.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test_clean.csv"))
    
    # Initialize predictor
    predictor = CreditCardFraudPredictor(model_dir)
    
    # Load models (replace with your timestamp)
    timestamp = "20241023_030219"
    predictor.load_models(timestamp)
    
    # Optimize thresholds
    print("Optimizing thresholds...")
    predictor.optimize_thresholds(X_test, y_test.label)
    
    # Make final predictions
    print("\nMaking final predictions...")
    predictions, probabilities = predictor.predict(X_test)
    
    # Save predictions and metrics
    output_path = os.path.join(current_dir, f'predictions_{timestamp}.csv')
    predictor.save_predictions(predictions, probabilities, output_path)
    predictor.save_model_metrics(timestamp)
    
    print(f"\nPredictions saved to: {output_path}")

if __name__ == "__main__":
    main()

# 3 models
# """
# import os
# import numpy as np
# import pandas as pd
# import pickle
# import json
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
# import matplotlib.pyplot as plt

# class CreditCardFraudPredictor:
#     def __init__(self, model_dir):
#         self.model_dir = model_dir
#         self.models = {}
#         self.predictions = {}
#         self.best_thresholds = {}
#         self.common_threshold = None
        
#     def load_models(self, timestamp):
#         """Load all trained models"""
#         # Load LightGBM
#         lgb_path = os.path.join(self.model_dir, f'lightgbm_model_{timestamp}.txt')
#         self.models['lgb'] = lgb.Booster(model_file=lgb_path)
        
#         # Load Random Forest
#         with open(os.path.join(self.model_dir, f'random_forest_model_{timestamp}.pkl'), 'rb') as f:
#             self.models['rf'] = pickle.load(f)
            
#         # Load XGBoost
#         with open(os.path.join(self.model_dir, f'xgb_random_search_model_{timestamp}.pkl'), 'rb') as f:
#             self.models['xgb'] = pickle.load(f)
    
#     def predict_probabilities(self, X):
#         """Get probability predictions from all models"""
#         self.predictions['lgb'] = self.models['lgb'].predict(X)
#         self.predictions['rf'] = np.mean([estimator.predict_proba(X)[:, 1] 
#                                         for estimator in self.models['rf'].estimators_], axis=0)
#         self.predictions['xgb'] = self.models['xgb'].predict_proba(X)[:, 1]
#         return self.predictions
    
#     def find_best_threshold(self, y_true, y_prob, thresholds):
#         """Find optimal threshold for binary classification"""
#         best_f1 = 0
#         best_threshold = 0
#         for threshold in thresholds:
#             y_pred = (y_prob >= threshold).astype(int)
#             f1 = f1_score(y_true, y_pred)
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold
#         return best_threshold, best_f1
    
#     def optimize_thresholds(self, X, y, threshold_range=np.arange(0.01, 1, 0.01)):
#         """Find optimal thresholds for all models"""
#         predictions = self.predict_probabilities(X)
        
#         # Find best threshold for each model
#         for model_name, y_prob in predictions.items():
#             best_threshold, best_f1 = self.find_best_threshold(y, y_prob, threshold_range)
#             self.best_thresholds[model_name] = {
#                 'threshold': best_threshold,
#                 'f1_score': best_f1
#             }
#             print(f"{model_name} - Best Threshold: {best_threshold:.2f}, Best F1: {best_f1:.4f}")
        
#         # Plot F1 scores vs thresholds
#         self.plot_threshold_optimization(y, threshold_range)
        
#         # Find common threshold
#         common_range = np.arange(0.3, 0.7, 0.01)
#         f1_scores_common = []
        
#         for threshold in common_range:
#             f1_scores = []
#             for y_prob in predictions.values():
#                 y_pred = (y_prob >= threshold).astype(int)
#                 f1_scores.append(f1_score(y, y_pred))
#             f1_scores_common.append(np.mean(f1_scores))
        
#         self.common_threshold = common_range[np.argmax(f1_scores_common)]
#         print(f"\nBest common threshold: {self.common_threshold:.2f}")
        
#         # Print results with common threshold
#         self.print_metrics_with_threshold(y, self.common_threshold)
    
#     def plot_threshold_optimization(self, y, thresholds):
#         """Plot F1 scores vs thresholds for all models"""
#         plt.figure(figsize=(12, 6))
#         colors = {'lgb': 'blue', 'xgb': 'orange', 'rf': 'green'}
        
#         for model_name, y_prob in self.predictions.items():
#             f1_scores = [f1_score(y, (y_prob >= t).astype(int)) for t in thresholds]
#             plt.plot(thresholds, f1_scores, label=model_name, color=colors[model_name])
            
#             # Add vertical lines for best thresholds
#             best_threshold = self.best_thresholds[model_name]['threshold']
#             plt.axvline(x=best_threshold, color=colors[model_name], 
#                        linestyle='--', label=f'{model_name} Best')
        
#         plt.title('F1 Score vs Threshold')
#         plt.xlabel('Threshold')
#         plt.ylabel('F1 Score')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
    
#     def print_metrics_with_threshold(self, y, threshold):
#         """Print precision, recall, and F1 score using specified threshold"""
#         print("\nMetrics with common threshold:")
#         for model_name, y_prob in self.predictions.items():
#             y_pred = (y_prob >= threshold).astype(int)
#             precision = precision_score(y, y_pred)
#             recall = recall_score(y, y_pred)
#             f1 = f1_score(y, y_pred)
#             print(f"{model_name} - Precision: {precision:.4f}, "
#                   f"Recall: {recall:.4f}, F1: {f1:.4f}")
    
#     def predict(self, X, use_common_threshold=True):
#         """Make final predictions using ensemble of models"""
#         predictions = self.predict_probabilities(X)
        
#         # Use weighted average for ensemble
#         weights = {'lgb': 0.4, 'rf': 0.3, 'xgb': 0.3}
#         ensemble_pred = sum(predictions[model] * weight 
#                           for model, weight in weights.items())
        
#         threshold = self.common_threshold if use_common_threshold else 0.5
#         return (ensemble_pred >= threshold).astype(int), ensemble_pred
    
#     def save_predictions(self, predictions, probabilities, output_path):
#         """Save predictions to CSV"""
#         pd.DataFrame({
#             'predicted_class': predictions,
#             'predicted_probability': probabilities
#         }).to_csv(output_path, index=False)

# def main():
#     # Set paths
#     current_dir = os.getcwd()
#     model_dir = os.path.join(current_dir, "models")
#     data_dir = os.path.join(current_dir, "training_data")
    
#     # Load test data
#     X_test = pd.read_csv(os.path.join(data_dir, "X_test_clean.csv"))
#     y_test = pd.read_csv(os.path.join(data_dir, "y_test_clean.csv"))
    
#     # Initialize predictor
#     predictor = CreditCardFraudPredictor(model_dir)
    
#     # Load models (replace with your timestamp)
#     timestamp = "20241023_023822"
#     predictor.load_models(timestamp)
    
#     # Optimize thresholds
#     print("Optimizing thresholds...")
#     predictor.optimize_thresholds(X_test, y_test.label)
    
#     # Make final predictions
#     print("\nMaking final predictions...")
#     predictions, probabilities = predictor.predict(X_test)
    
#     # Save predictions
#     output_path = os.path.join(current_dir, f'predictions_{timestamp}.csv')
#     predictor.save_predictions(predictions, probabilities, output_path)
#     print(f"\nPredictions saved to: {output_path}")

# if __name__ == "__main__":
#     main()
# """