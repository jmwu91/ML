import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class FeatureEngineering:
    def __init__(self):
        self.label_encoders = {}
        self.feature_maps = {}
        
    def read_data(self, train_path, test_path):
        """讀取訓練和測試數據"""
        try:
            df_raw_train = pd.read_csv(train_path)
            df_raw_test = pd.read_csv(test_path)
            print(f"原始訓練資料: {df_raw_train.shape}")
            print(f"測試資料: {df_raw_test.shape}")
            return df_raw_train, df_raw_test
        except Exception as e:
            print(f"讀取資料時發生錯誤: {str(e)}")
            raise

    def drop_unused_columns(self, df):
        """移除不需要的列"""
        return df.drop(['bnsfg', 'iterm', 'flbmk', 'insfg', 'flam1'], axis=1)

    def create_features(self, df, is_training=True):
        """創建新特徵"""
        df = df.copy()
        
        # 計算當次消費跟所有過去平均消費之間的差異
        df['avg_conam'] = df.groupby('cano')['conam'].transform('mean')
        df['diff_conam'] = df['conam'] - df['avg_conam']
        
        # 計算當次刷卡城市是否是第一次出現
        df['first_appearance'] = df.groupby(['cano', 'scity'])['locdt'].transform('min')
        df['is_city_first_appearance'] = df['locdt'] == df['first_appearance']
        
        # 計算當次消費幣別是否是第一次出現
        df['first_appearance'] = df.groupby(['cano', 'csmcu'])['locdt'].transform('min')
        df['is_csmcu_first_appearance'] = df['locdt'] == df['first_appearance']
        
        # 計算總刷卡次數和每日的刷卡次數
        df['cano_count'] = df.groupby('cano')['cano'].transform('count')
        df['Is_First_Occurrence'] = ~df['cano'].duplicated()
        df['freq_perday'] = df.groupby(['cano', 'locdt'])['cano'].transform('count')
        
        # 移除中間產物
        df = df.drop(['avg_conam', 'first_appearance'], axis=1)
        
        return df

    def encode_categorical_features(self, df_train, df_test):
        """對類別特徵進行編碼"""
        categorical_features = ['mchno', 'acqic', 'cano']
        
        for feature in categorical_features:
            le = LabelEncoder()
            # 對訓練資料進行fit和transform
            if feature in df_train.columns:
                le.fit(df_train[feature].fillna('MISSING'))
                df_train[feature] = le.transform(df_train[feature].fillna('MISSING'))
                # 儲存mapping供測試資料使用
                self.feature_maps[feature] = dict(zip(le.classes_, le.transform(le.classes_)))
                
                # 對測試資料進行transform，對於未見過的值使用特殊值
                df_test[feature] = df_test[feature].map(self.feature_maps[feature])
                df_test[feature] = df_test[feature].fillna(len(self.feature_maps[feature]))
                
        return df_train, df_test

    def clean_data(self, df):
        """清理數據中的缺失值"""
        df = df.copy()
        
        # 使用固定值填充
        fill_values = {
            'etymd': 99,
            'stscd': 99,
            'stocn': 999,
            'hcefg': 999,
            'csmcu': 999,
            'scity': 999,
            'mcc': 999
        }
        
        for col, value in fill_values.items():
            df[col] = df[col].fillna(value)
        
        # 使用前向/後向填充
        df['loctm'] = df['loctm'].fillna(method='ffill')
        df['contp'] = df['contp'].fillna(method='ffill')
        df['conam'] = df['conam'].fillna(method='bfill')
        df['ecfg'] = df['ecfg'].fillna(method='bfill')
        df['ovrlt'] = df['ovrlt'].fillna(method='ffill')
        df['csmam'] = df['csmam'].fillna(method='bfill')
        df['flg_3dsmk'] = df['flg_3dsmk'].fillna(method='ffill')
        
        # 特殊處理
        df['diff_conam'] = df['diff_conam'].fillna(0)
        df['cano_count'] = df['cano_count'].fillna(method='bfill')
        df['freq_perday'] = df['freq_perday'].fillna(method='bfill')
        
        return df

    def prepare_training_data(self, df_raw_train):
        """準備訓練數據"""
        X = df_raw_train.drop(['txkey', 'locdt', 'label', 'chid'], axis=1)
        y = df_raw_train['label']
        
        # 重採樣
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)
        
        # 將數值型特徵轉換為對應的類型
        X['conam'] = X['conam'].astype('float')
        X['csmam'] = X['csmam'].astype('float')
        X['loctm'] = X['loctm'].astype('int')
        X['cano_count'] = X['cano_count'].astype('int')
        X['diff_conam'] = X['diff_conam'].astype('float')
        X['freq_perday'] = X['freq_perday'].astype('int')
        
        X_res, y_res = pipeline.fit_resample(X, y)
        
        return X_res, y_res

    def prepare_test_data(self, df_raw_test):
        """準備測試數據"""
        X_test = df_raw_test.drop(['txkey', 'locdt', 'chid'], axis=1)
        
        # 將數值型特徵轉換為對應的類型
        X_test['conam'] = X_test['conam'].astype('float')
        X_test['csmam'] = X_test['csmam'].astype('float')
        X_test['loctm'] = X_test['loctm'].astype('int')
        X_test['cano_count'] = X_test['cano_count'].astype('int')
        X_test['diff_conam'] = X_test['diff_conam'].astype('float')
        X_test['freq_perday'] = X_test['freq_perday'].astype('int')
        
        return X_test

    def process_data(self, train_path, test_path):
        """完整的數據處理流程"""
        # 讀取數據
        print("讀取數據...")
        df_raw_train, df_raw_test = self.read_data(train_path, test_path)
        
        # 移除不需要的列
        print("移除不需要的列...")
        df_raw_train = self.drop_unused_columns(df_raw_train)
        df_raw_test = self.drop_unused_columns(df_raw_test)
        
        # 創建特徵
        print("創建新特徵...")
        df_raw_train = self.create_features(df_raw_train, is_training=True)
        df_raw_test = self.create_features(df_raw_test, is_training=False)
        
        # 編碼類別特徵
        print("編碼類別特徵...")
        df_raw_train, df_raw_test = self.encode_categorical_features(df_raw_train, df_raw_test)
        
        # 清理數據
        print("清理數據...")
        df_raw_train = self.clean_data(df_raw_train)
        df_raw_test = self.clean_data(df_raw_test)
        
        # 準備訓練和測試數據
        print("準備最終數據集...")
        X_res, y_res = self.prepare_training_data(df_raw_train)
        X_test = self.prepare_test_data(df_raw_test)
        
        return X_res, y_res, X_test

def main():
    # 設定路徑
    current_dir = os.getcwd()
    train_path = os.path.join(current_dir, "train.csv")
    test_path = os.path.join(current_dir, "X_test.csv")
    
    # 創建特徵工程實例
    fe = FeatureEngineering()
    
    try:
        # 處理數據
        print("開始處理數據...")
        X_res, y_res, X_test = fe.process_data(train_path, test_path)
        
        # 建立training_data目錄
        output_dir = os.path.join(current_dir, "training_data")
        os.makedirs(output_dir, exist_ok=True)
        
        # 分割成訓練集和驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            X_res, 
            y_res, 
            test_size=0.1, 
            random_state=20231123, 
            stratify=y_res
        )
        
        # 保存處理後的數據，使用原始的檔案名稱
        print("\n保存處理後的數據...")
        X_train.to_csv(os.path.join(output_dir, "X_train_clean.csv"), index=False)
        X_val.to_csv(os.path.join(output_dir, "X_test_clean.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train_clean.csv"), index=False)
        pd.DataFrame(y_val).to_csv(os.path.join(output_dir, "y_test_clean.csv"), index=False)
        
        # 保存處理後的測試數據
        print("保存處理後的測試數據...")
        X_test.to_csv("X_test_pub.csv", index=False)
        
        print("\n數據處理完成！")
        print(f"訓練集形狀: {X_train.shape}")
        print(f"驗證集形狀: {X_val.shape}")
        print(f"測試集形狀: {X_test.shape}")
        
        # 打印類別分布
        print("\n訓練集標籤分布:")
        print(pd.Series(y_train).value_counts(normalize=True))
        print("\n驗證集標籤分布:")
        print(pd.Series(y_val).value_counts(normalize=True))
        
    except Exception as e:
        print(f"處理過程中發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()