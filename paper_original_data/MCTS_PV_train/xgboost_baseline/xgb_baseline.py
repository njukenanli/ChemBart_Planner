import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import time

class XGBValueBaseline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.n_bits = 2048
        self.model_v = None
        
    def extract_ecfp(self, smiles, radius=2):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        for idx in fp.GetOnBits():
            arr[idx] = 1.0
        return arr

    def load_and_process(self):
        print("Loading and processing data...")
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
            
        X_list = []
        Y_v_list = []
        
        count = 0
        for item in raw_data:
            try:
                raw_smiles = item[0][0]
                smiles = raw_smiles[7:]
                ecfp = self.extract_ecfp(smiles)
                if ecfp is None: continue
                
                v_val = float(item[1][0])
                X_list.append(ecfp)
                Y_v_list.append(v_val)
                count += 1
            except Exception as e:
                continue
        
        print(f"Total valid samples loaded: {count}")
        return np.array(X_list), np.array(Y_v_list)

    def train(self, use_gpu=True, n_estimators=500, max_depth=6, learning_rate=0.05):
        X, Y_v = self.load_and_process()
        if len(X) == 0:
            print("No data found!")
            return

        total_samples = len(X)
        
        # --- 核心逻辑：10/12, 1/12, 1/12 划分 ---
        # 第一步：先切分出 Test (1/12)
        # 剩余部分比例为 11/12
        test_size_ratio = 1 / 12
        remaining_ratio = 1 - test_size_ratio # 11/12
        
        X_temp, X_test, Y_temp, Y_test = train_test_split(
            X, Y_v, 
            test_size=test_size_ratio, 
            random_state=42, 
            shuffle=True
        )
        
        # 第二步：从剩余部分 (11/12) 中切分出 Validation (1/12)
        # 此时 X_temp 占总数的 11/12。我们需要从中取出 1/11 的比例，才能使得最终 Validation 占总数的 1/12。
        # 计算：(11/12) * (1/11) = 1/12
        # 剩下的就是 Train: (11/12) * (10/11) = 10/12
        val_size_ratio_from_temp = (1/12) / remaining_ratio # 即 1/11
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_temp, Y_temp,
            test_size=val_size_ratio_from_temp,
            random_state=42,
            shuffle=True
        )
        
        print(f"\nData Split Summary (Target 10:1:1):")
        print(f"  Total: {total_samples}")
        print(f"  Train:      {len(X_train):6d} ({len(X_train)/total_samples:.2%})")
        print(f"  Validation: {len(X_val):6d} ({len(X_val)/total_samples:.2%})")
        print(f"  Test:       {len(X_test):6d} ({len(X_test)/total_samples:.2%})")
        
        device_param = "cuda" if use_gpu else "cpu"
        print(f"Using device: {device_param}")

        # 定义模型
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device=device_param,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20
        )
        
        print("\nStart Training (Monitoring Validation Set for Early Stopping)...")
        start_time = time.time()
        
        # 训练
        # eval_set 顺序很重要：[(Train, TrainLabel), (Val, ValLabel)]
        # XGBoost 会使用最后一个集合 (Validation) 来进行早停判断
        model.fit(
            X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)],
            verbose=50 
        )
        
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds.")
        
        self.model_v = model
        
        # 评估所有三个集合
        self.detailed_evaluation(X_train, Y_train, X_val, Y_val, X_test, Y_test)
        
        joblib.dump(self.model_v, 'xgb_value_model_1011.pkl')
        print("\nModel saved to 'xgb_value_model_1011.pkl'")

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
            print("Warning: Significant Overfitting detected.")
        else:
            print("Generalization looks good.")
            
        # 最佳迭代次数
        best_iter = self.model_v.best_iteration
        if best_iter:
            print(f"\n  Best Iteration (stopped at): {best_iter}")
        else:
            print(f"\n  Completed full iterations without early stopping.")

if __name__ == "__main__":
    DATA_PATH = '../ChemBart/data/filtered_MCTS_data.json'
    
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except:
        use_gpu = False
        
    model = XGBValueBaseline(DATA_PATH)
    # 可以根据需要调整参数
    model.train(use_gpu=use_gpu, n_estimators=500, max_depth=12, learning_rate=0.05)