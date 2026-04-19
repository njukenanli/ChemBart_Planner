import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import time

class RFValueBaseline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model_v = None 
        self.n_bits = 2048 # ECFP 长度
        
    def extract_ecfp(self, smiles, radius=2):
        """生成 ECFP 指纹并转为 numpy 数组"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        return arr

    def load_and_process(self):
        """加载数据，仅提取 V 作为主要标签"""
        print("Loading data for Random Forest Value Network...")
        
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
            
        X_list = []
        Y_v_list = []
        valid_count = 0
        
        for item in raw_data:
            try:
                # 1. 解析 SMILES
                
                raw_smiles = item[0][0]
                smiles = raw_smiles[7:]

                # 2. 生成特征 (ECFP)
                ecfp = self.extract_ecfp(smiles)
                if ecfp is None:
                    continue
                
                # 3. 提取价值标签 V
                v_val = float(item[1][0])
                
                X_list.append(ecfp)
                Y_v_list.append(v_val)
                valid_count += 1
                
            except Exception as e:
                continue
        
        print(f"Processed {valid_count} valid samples for Value prediction.")
        
        X = np.array(X_list)
        Y_v = np.array(Y_v_list)
        
        return X, Y_v

    def train(self, n_estimators=200, max_depth=None, min_samples_leaf=2, n_jobs=-1):
        """训练随机森林价值模型 (10/12 Train, 1/12 Val, 1/12 Test)"""
        X, Y_v = self.load_and_process()
        
        if len(X) == 0:
            print("No data to train!")
            return

        total_samples = len(X)
        
        # --- 核心逻辑：10/12, 1/12, 1/12 划分 ---
        # 第一步：切分 Test (1/12)
        test_ratio = 1 / 12
        remaining_ratio = 1 - test_ratio # 11/12
        
        X_temp, X_test, Y_temp, Y_test = train_test_split(
            X, Y_v, 
            test_size=test_ratio, 
            random_state=42, 
            shuffle=True
        )
        
        # 第二步：从剩余部分切分 Validation (目标是总量的 1/12)
        # 计算比例：(1/12) / (11/12) = 1/11
        val_ratio_from_temp = (1/12) / remaining_ratio
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_temp, Y_temp,
            test_size=val_ratio_from_temp,
            random_state=42,
            shuffle=True
        )
        
        print(f"\nData Split Summary (Target 10:1:1):")
        print(f"  Total: {total_samples}")
        print(f"  Train:      {len(X_train):6d} ({len(X_train)/total_samples:.2%})")
        print(f"  Validation: {len(X_val):6d} ({len(X_val)/total_samples:.2%})")
        print(f"  Test:       {len(X_test):6d} ({len(X_test)/total_samples:.2%})")
        
        print(f"\nTraining Random Forest Regressor (Value Net)...")
        print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")
        
        start_time = time.time()
        
        # 定义模型
        self.model_v = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=n_jobs, # 使用所有 CPU 核心
            verbose=1
        )
        
        self.model_v.fit(X_train, Y_train)
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds.")
        
        # 评估所有三个集合
        self.detailed_evaluation(X_train, Y_train, X_val, Y_val, X_test, Y_test)
        
        # 保存模型
        joblib.dump(self.model_v, 'rf_value_model_1011.pkl')
        print("\nModel saved to 'rf_value_model_1011.pkl'")
        
        # 输出特征重要性
        self.print_feature_importance(top_n=10)

    def detailed_evaluation(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        print("\n" + "="*50)
        print("           FINAL EVALUATION RESULTS          ")
        print("="*50)
        print(f"{'Dataset':<12} | {'Samples':<8} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10}")
        print("-" * 58)
        
        results = []
        
        # 1. Train
        y_train_pred = self.model_v.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(Y_train, y_train_pred))
        train_mae = mean_absolute_error(Y_train, y_train_pred)
        train_r2 = r2_score(Y_train, y_train_pred)
        results.append(("Train", len(X_train), train_rmse, train_mae, train_r2))
        
        # 2. Validation
        y_val_pred = self.model_v.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(Y_val, y_val_pred))
        val_mae = mean_absolute_error(Y_val, y_val_pred)
        val_r2 = r2_score(Y_val, y_val_pred)
        results.append(("Validation", len(X_val), val_rmse, val_mae, val_r2))
        
        # 3. Test (最终报告指标)
        y_test_pred = self.model_v.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
        test_mae = mean_absolute_error(Y_test, y_test_pred)
        test_r2 = r2_score(Y_test, y_test_pred)
        results.append(("Test", len(X_test), test_rmse, test_mae, test_r2))
        
        for name, count, rmse, mae, r2 in results:
            print(f"{name:<12} | {count:<8} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f}")
            
        print("-" * 58)
        
        # 过拟合分析 (Train vs Val)
        gap = val_rmse - train_rmse
        gap_ratio = (gap / train_rmse) * 100 if train_rmse > 0 else 0
        print(f"\nOverfitting Check (Val vs Train):")
        print(f"  RMSE Gap: {gap:.4f} ({gap_ratio:.2f}%)")
        
        if gap_ratio > 20:
            print("  ⚠️  Warning: Significant Overfitting detected.")
            print("     Suggestion: Increase min_samples_leaf or reduce max_depth.")
        elif gap_ratio < 0:
            print("  ℹ️  Note: Val error is lower than Train error (lucky split or regularization working well).")
        else:
            print("  ✅ Generalization looks good.")

    def print_feature_importance(self, top_n=10):
        """打印最重要的 ECFP 位点"""
        if self.model_v is None:
            return
            
        importances = self.model_v.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\nTop {top_n} Important ECFP Bits:")
        for i in range(top_n):
            idx = indices[i]
            print(f"  Bit {idx}: Importance = {importances[idx]:.4f}")

if __name__ == "__main__":
    # 请确保路径正确
    DATA_PATH = '../ChemBart/data/filtered_MCTS_data.json'
    
    rf = RFValueBaseline(DATA_PATH)
    
    # 训练参数建议
    # n_estimators=200: 稳定
    # max_depth=None: 让树充分生长 (RF 通常靠 min_samples_leaf 控制过拟合)
    # min_samples_leaf=2: 防止对单个噪声点过拟合
    rf.train(n_estimators=200, max_depth=None, min_samples_leaf=2, n_jobs=-1)