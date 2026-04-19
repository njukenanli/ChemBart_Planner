import json
import numpy as np
import deepchem as dc
from deepchem.models import GCNModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

class DeepChemGCNBaseline:
    def __init__(self, data_path):
        self.data_path = data_path
        # DeepChem 的 MolGraphConvFeaturizer 会自动处理原子特征，无需手动定义长度
        self.featurizer = dc.feat.MolGraphConvFeaturizer() 

    def load_and_process(self):
        """加载数据，返回 (graph_objects, values, smiles_list)"""
        print("Loading and featurizing data with DeepChem...")
        
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
            
        valid_graphs = []
        valid_values = []
        valid_smiles = []
        
        count = 0
        for item in raw_data:
            try:
                # 1. 解析 SMILES
                raw_smiles = item[0][0]
                smiles = raw_smiles[7:]
                
                # 2. 提取 Value
                v_val = float(item[1][0])
                
                # 3. 使用 DeepChem Featurizer 转换
                # featurize 返回一个列表，我们取第一个元素（因为一次处理一个分子）
                graph_obj = self.featurizer.featurize([smiles])[0]
                
                if graph_obj is None:
                    continue
                
                valid_graphs.append(graph_obj)
                valid_values.append(v_val)
                valid_smiles.append(smiles)
                count += 1
                
                if count % 5000 == 0:
                    print(f"  Processed {count} samples...")
                    
            except Exception as e:
                # print(f"Error processing {item[0][0]}: {e}")
                continue
        
        print(f"Total valid graphs: {len(valid_graphs)}")
        return valid_graphs, np.array(valid_values), valid_smiles

    def train(self, n_epochs=100, batch_size=64, learning_rate=0.001, 
              graph_conv_layers=[64, 64], dropout=0.2):
        
        graphs, values, smiles = self.load_and_process()
        if len(graphs) == 0:
            print("No valid data!")
            return

        total_samples = len(graphs)
        
        # --- 10/12, 1/12, 1/12 划分 ---
        # 注意：DeepChem 的 Dataset 可以直接接受列表切片
        test_ratio = 1 / 12
        remaining_ratio = 1 - test_ratio
        val_ratio_from_temp = (1/12) / remaining_ratio
        
        # 先打乱索引以保证随机性 (DeepChem 的 split 也可以，但这里用 sklearn 更直观控制比例)
        indices = np.arange(total_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # 重排数据
        graphs_shuffled = [graphs[i] for i in indices]
        values_shuffled = values[indices]
        
        # 切分 Test
        n_test = int(total_samples * test_ratio)
        graphs_temp, graphs_test = graphs_shuffled[:-n_test], graphs_shuffled[-n_test:]
        y_temp, y_test = values_shuffled[:-n_test], values_shuffled[-n_test:]
        
        # 切分 Val
        n_val = int(len(graphs_temp) * val_ratio_from_temp)
        graphs_train, graphs_val = graphs_temp[:-n_val], graphs_temp[-n_val:]
        y_train, y_val = y_temp[:-n_val], y_temp[-n_val:]
        
        print(f"\nData Split Summary:")
        print(f"  Train:      {len(graphs_train):6d} ({len(graphs_train)/total_samples:.2%})")
        print(f"  Validation: {len(graphs_val):6d} ({len(graphs_val)/total_samples:.2%})")
        print(f"  Test:       {len(graphs_test):6d} ({len(graphs_test)/total_samples:.2%})")

        # 构建 DeepChem Dataset
        # X 是图对象列表，y 是标签
        train_dataset = dc.data.NumpyDataset(X=np.array(graphs_train, dtype=object), y=y_train.reshape(-1, 1))
        val_dataset = dc.data.NumpyDataset(X=np.array(graphs_val, dtype=object), y=y_val.reshape(-1, 1))
        test_dataset = dc.data.NumpyDataset(X=np.array(graphs_test, dtype=object), y=y_test.reshape(-1, 1))

        print("\nInitializing GCNModel...")
        # 初始化模型
        # mode='regression' 用于预测连续值 V
        # number_atom_features: MolGraphConvFeaturizer 默认生成约 30-40 维特征，模型会自动适配或需指定
        # 这里让模型自动推断或使用默认值，通常 Featurizer 输出的维度是固定的
        model = GCNModel(
            n_tasks=1,
            mode='regression',
            graph_conv_layers=graph_conv_layers,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            number_atom_features=30, # Featurizer 默认输出维度，若报错可调整为 featurizer.feature_dim()
            self_loop=True
        )

        print(f"Start Training for {n_epochs} epochs...")
        start_time = time.time()
        
        # 训练
        # DeepChem 的 fit 返回平均 loss
        losses = []
        for epoch in range(n_epochs):
            loss = model.fit(train_dataset, nb_epoch=1) # 每次 fit 一个 epoch 以便监控
            losses.append(loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")
                
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds.")

        # 评估
        self.evaluate(model, train_dataset, val_dataset, test_dataset)

    def evaluate(self, model, train_ds, val_ds, test_ds):
        print("\n" + "="*50)
        print("           FINAL EVALUATION RESULTS          ")
        print("="*50)
        print(f"{'Dataset':<12} | {'Samples':<8} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10}")
        print("-" * 58)
        
        results = []
        datasets = [('Train', train_ds), ('Validation', val_ds), ('Test', test_ds)]
        
        for name, ds in datasets:
            # 预测
            # model.predict 返回形状为 (N, 1) 的数组
            preds = model.predict(ds).reshape(-1)
            true_vals = ds.y.reshape(-1)
            
            rmse = np.sqrt(mean_squared_error(true_vals, preds))
            mae = mean_absolute_error(true_vals, preds)
            r2 = r2_score(true_vals, preds)
            
            results.append((name, len(ds), rmse, mae, r2))
            
        for name, count, rmse, mae, r2 in results:
            print(f"{name:<12} | {count:<8} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f}")
            
        print("-" * 58)
        
        # 简单过拟合检查
        train_rmse = results[0][2]
        val_rmse = results[1][2]
        gap = val_rmse - train_rmse
        gap_ratio = (gap / train_rmse) * 100 if train_rmse > 0 else 0
        
        print(f"\nOverfitting Check (Val vs Train):")
        print(f"  RMSE Gap: {gap:.4f} ({gap_ratio:.2f}%)")
        if gap_ratio > 20:
            print("Warning: Possible Overfitting. Try increasing dropout or reducing layers.")
        else:
            print("Generalization looks good.")

if __name__ == "__main__":
    DATA_PATH = '../ChemBart/data/filtered_MCTS_data.json'
    
    gcn = DeepChemGCNBaseline(DATA_PATH)
    
    # 超参数建议
    # graph_conv_layers: [64, 64] 是经典配置，也可尝试 [128, 128]
    # dropout: 0.2 或 0.3 防止过拟合
    # learning_rate: 1e-3 是常用起点
    gcn.train(
        n_epochs=200, 
        batch_size=64, 
        learning_rate=0.001, 
        graph_conv_layers=[64, 64], 
        dropout=0.2
    )