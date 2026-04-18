# 网格搜索
from ChemBart import CB_Regression
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import random
import os

# ================= config=================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LABEL_NUM = 1
DATA_PATH = "data/yield/bh.json"
MODEL_NAME_BASE = "Buchwald-Hartwig_Opt"
PRE_MODEL = 'ChemBart_MIT_9'
DEVICE = "cuda:3"


#LR_CANDIDATES = [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
LR_CANDIDATES = [5e-5, 1e-4, 1e-3]
DECAY_CANDIDATES = [1e-4, 1e-5, 1e-6, 0.0] 

# fixed config
EPOCHS = 500  
BATCH_SIZE = 8
ID_MAXLEN = 512
TRAIN_PROP = 0.6
VALID_PROP = 0.1

# ================= dataload =================
task_token = ["<n00>", "<n01>", "<n02>", "<n03>", "<n04>"]
ends = "".join(task_token[:LABEL_NUM - 1]) + "<end>"

print("loading data...")
with open(DATA_PATH, "r", encoding='utf-8') as f:
    data_list = json.load(f)


for i in range(len(data_list)):
    data_list[i][0] += ends

random.shuffle(data_list)

total_size = len(data_list)
train_size = int(TRAIN_PROP * total_size)
valid_size = int(VALID_PROP * total_size)
test_size = total_size - train_size - valid_size

print(f"数据总量：{total_size}, Train: {train_size}, Valid: {valid_size}, Test: {test_size}")


raw_labels = [item[1] for item in data_list]
train_raw_labels = raw_labels[:train_size]

global_mean = np.mean(train_raw_labels)
global_std = np.std(train_raw_labels)
if global_std < 1e-6: global_std = 1e-6

print(f"Global Mean: {global_mean:.4f}, Std: {global_std:.4f}")

def prepare_data_for_model(data_slice, mean, std):
    processed = []
    for item in data_slice:
        original_label = item[1]
        norm_label = (original_label - mean) / std
        processed.append([item[0], [float(norm_label)]])
    return processed


train_data_norm = prepare_data_for_model(data_list[:train_size], global_mean, global_std)
valid_data_norm = prepare_data_for_model(data_list[train_size:train_size+valid_size], global_mean, global_std)
test_data_norm = prepare_data_for_model(data_list[train_size+valid_size:], global_mean, global_std)

combined_train_val = train_data_norm + valid_data_norm

# =================grid search =================
best_valid_rmse = float('inf')
best_params = {'lr': None, 'decay': None}
best_model_path = None

print("\n" + "="*50)
print("starting grid search...")
print("="*50)

results_log = []

for lr in LR_CANDIDATES:
    for decay in DECAY_CANDIDATES:
        print(f"\n>>> try：lr={lr}, decay={decay}")
        

        temp_name = f"{MODEL_NAME_BASE}_lr{lr}_dc{decay}"
        model = CB_Regression(
            name=temp_name, 
            pre_model=PRE_MODEL, 
            label_num=LABEL_NUM, 
            device=DEVICE
        )
        
        try:

            model.fit(
                data=combined_train_val,
                epoch=EPOCHS,
                batch_size=BATCH_SIZE,
                tr=train_size,
                val=valid_size,
                te=test_size, 
                lr=lr,
                decay=decay,
                id_maxlen=ID_MAXLEN
            )
            
            val_rmse_list, _ = model.test(valid_data_norm, batch_size=BATCH_SIZE)
            current_val_rmse = val_rmse_list[0] # 假设只有一个回归任务
            
            print(f"    Valid RMSE: {current_val_rmse:.4f}")
            
            results_log.append({
                'lr': lr, 'decay': decay, 'val_rmse': current_val_rmse
            })
            

            if current_val_rmse < best_valid_rmse:
                best_valid_rmse = current_val_rmse
                best_params = {'lr': lr, 'decay': decay}
                best_model_path = model.save_path 
                print(f"    >>> Best！RMSE: {best_valid_rmse:.4f}")
            else:
                # 如果不是最佳，可以选择删除临时模型文件以节省空间，或者保留
                pass
                
        except Exception as e:
            print(f"    error：{e}")
            continue
        finally:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ================= output =================
print("\n" + "="*50)
print("finish grid search！")
print("="*50)
print(f"best learning rate (lr): {best_params['lr']}")
print(f"best weight decay (decay): {best_params['decay']}")
print(f"best validation RMSE: {best_valid_rmse:.4f}")



# =================   TEST   =================
print("\nTEST")
LR_CANDIDATES = [5e-5, 1e-4, 1e-3]
DECAY_CANDIDATES = [1e-4, 1e-5, 1e-6, 0.0]

for lr in LR_CANDIDATES:
    for decay in DECAY_CANDIDATES:
        temp_name = f"{MODEL_NAME_BASE}_lr{lr}_dc{decay}"
        final_model = CB_Regression(
            name=temp_name,
            pre_model=PRE_MODEL,
            label_num=LABEL_NUM,
            device=DEVICE
        )


        (RMSE_list, reslist) = final_model.test(test_data_norm, batch_size=4)


        y_pred, y_test = [], []


        if reslist and len(reslist) > 0:
            task_res = reslist[0]
            for item in task_res:
                if isinstance(item, tuple) and len(item) == 2:

                    p_val = (item[0] * global_std + global_mean) * 100
                    t_val = (item[1] * global_std + global_mean) * 100
                    
                    y_pred.append(p_val)
                    y_test.append(t_val)

        y_pred = np.clip(y_pred, 0, 100) 
        y_test = np.array(y_test)

        final_r2 = r2_score(y_test, y_pred)
        final_mae = mean_absolute_error(y_test, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n=== result (Best Params: lr={best_params['lr']}, decay={best_params['decay']}) ===")
        print(f"R² Score: {final_r2:.4f}")
        print(f"MAE: {final_mae:.4f}")
        print(f"RMSE: {final_rmse:.4f}")


        try:
            plot = make_plot(y_test, y_pred, f'Buchwald-Hartwig (Optimized)\nR²={final_r2:.3f}')
        except NameError:

            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
            plt.plot([0, 100], [0, 100], 'r--', lw=2)
            plt.xlabel('Experimental Yield (%)')
            plt.ylabel('Predicted Yield (%)')
            plt.title(f'Buchwald-Hartwig Yield Prediction\nR² = {final_r2:.4f}, MAE = {final_mae:.2f}%')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'BH_yield_gridsearch/{temp_name}.png', dpi=300)

