# 网格搜索
from ChemBart import CB_Regression
import json
import numpy as np
import random

# ================= config=================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LABEL_NUM = 1
DATA_PATH = "data/yield/bh_C_N_yield/MFF_Test2/train.json"
MODEL_NAME_BASE = "Buchwald-Hartwig_test2"
PRE_MODEL = 'ChemBart_MIT_9'
DEVICE = "cuda:0"

# grid search config
LR_CANDIDATES = [5e-5, 1e-5, 5e-6, 1e-6]
DECAY_CANDIDATES = [1e-4, 1e-5, 1e-6] 

# fixed config
EPOCHS = 500  
BATCH_SIZE = 4
ID_MAXLEN = 400
TRAIN_PROP = 0.9
VALID_PROP = 0.1

# ================= dataload =================
task_token = ["<n00>", "<n01>", "<n02>", "<n03>", "<n04>"]
ends = "".join(task_token[:LABEL_NUM - 1]) + "<end>"

print("loading data...")
with open(DATA_PATH, "r", encoding='utf-8') as f:
    data_list = json.load(f)


for i in range(len(data_list)):
    data_list[i][0] += ends

# Shuffle 
random.shuffle(data_list)

total_size = len(data_list)
train_size = int(TRAIN_PROP * total_size)
valid_size = int(VALID_PROP * total_size)
test_size = total_size - train_size - valid_size

print(f"Total:{total_size}, Train: {train_size}, Valid: {valid_size}, Test: {test_size}")


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

# ================= grid search =================
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
                te=test_size, # 
                lr=lr,
                decay=decay,
                id_maxlen=ID_MAXLEN
            )
            
            val_rmse_list, _ = model.test(valid_data_norm, batch_size=BATCH_SIZE)
            current_val_rmse = val_rmse_list[0] 
            
            print(f"    验证集 RMSE: {current_val_rmse:.4f}")
            
            results_log.append({
                'lr': lr, 'decay': decay, 'val_rmse': current_val_rmse
            })
            
            if current_val_rmse < best_valid_rmse:
                best_valid_rmse = current_val_rmse
                best_params = {'lr': lr, 'decay': decay}
                best_model_path = model.save_path
                print(f"    >>> Best！RMSE: {best_valid_rmse:.4f}")
            else:
                pass
                
        except Exception as e:
            print(f"   error：{e}")
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

