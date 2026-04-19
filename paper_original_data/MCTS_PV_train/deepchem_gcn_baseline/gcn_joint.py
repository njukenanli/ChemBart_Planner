import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from dgllife.model import GCNPredictor
from dgllife.utils import smiles_to_bigraph
from tqdm import tqdm
import time
import os
import dgl

# ================= 配置部分 =================
DATA_PATH = '../ChemBart/data/filtered_MCTS_data.json'
ACTION_SPACE_SIZE = 13312
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
N_EPOCHS = 200
HIDDEN_FEATS = [64, 64]
DROPOUT = 0.2
PATIENCE = 10
LAMBDA_POLICY = 1.0
LAMBDA_VALUE = 1.0
# ===========================================

print(f"🚀 Using Device: {DEVICE}")

# --- 自定义特征提取 ---
def simple_atom_featurizer(atom):
    symbol = atom.GetSymbol()
    # 去重后的原子类型列表
    atom_types = ['C', 'O', 'N', 'F', 'Cl', 'S', 'Br', 'P', 'B', 'I', 'Si']
    if symbol in atom_types:
        vec = [1.0 if symbol == t else 0.0 for t in atom_types]
        vec.append(0.0) # Other
    else:
        vec = [0.0] * len(atom_types)
        vec.append(1.0)
    return torch.tensor(vec, dtype=torch.float32)

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        g = smiles_to_bigraph(smiles, node_featurizer=None)
        if g is None: return None
        atom_feats = [simple_atom_featurizer(atom) for atom in mol.GetAtoms()]
        g.ndata['h'] = torch.stack(atom_feats, dim=0)
        return g
    except Exception:
        return None

# --- 数据集类 ---
class JointMoleculeDataset(Dataset):
    def __init__(self, data_items):
        self.graphs = []
        self.labels_p = []
        self.labels_v = []
        
        print("Converting SMILES to Graphs & Processing Labels...")
        count = 0
        for item in tqdm(data_items):
            try:
                raw_smiles = item[0][0]
                smiles = raw_smiles[7:] if len(raw_smiles) > 7 else raw_smiles
                
                g = mol_to_graph(smiles)
                if g is None: continue
                
                v_val = float(item[1][0])
                
                p_vec = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
                indices = item[2]
                probs = item[1][1]
                
                if len(indices) > 0 and max(indices) >= ACTION_SPACE_SIZE:
                    continue
                
                for idx, prob in zip(indices, probs):
                    p_vec[idx] = prob
                
                self.graphs.append(g)
                self.labels_p.append(p_vec)
                self.labels_v.append(v_val)
                count += 1
            except Exception:
                continue
        
        print(f"✅ Valid samples: {count}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels_p[idx], torch.tensor([self.labels_v[idx]], dtype=torch.float32)

def collate_fn_joint(batch):
    graphs, labels_p, labels_v = map(list, zip(*batch))
    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    
    if not valid_indices:
        return None, None, None
    
    batch_graphs = [graphs[i] for i in valid_indices]
    batch_p = torch.stack([labels_p[i] for i in valid_indices]).to(DEVICE)
    batch_v = torch.cat([labels_v[i] for i in valid_indices], dim=0).to(DEVICE)
    
    batched_graph = dgl.batch(batch_graphs).to(DEVICE)
    return batched_graph, batch_p, batch_v

# --- 联合模型定义 ---
class JointGCNModel(nn.Module):
    def __init__(self, node_feat_dim, action_size):
        super(JointGCNModel, self).__init__()
        self.backbone = GCNPredictor(
            in_feats=node_feat_dim,
            hidden_feats=HIDDEN_FEATS,
            activation=[nn.ReLU()] * len(HIDDEN_FEATS),
            residual=[True] * len(HIDDEN_FEATS),
            batchnorm=[True] * len(HIDDEN_FEATS),
            dropout=[DROPOUT] * len(HIDDEN_FEATS),
            n_tasks=1,
            predictor_hidden_feats=128,
            predictor_dropout=DROPOUT
        )
        
        self.encoder = self.backbone.gnn
        self.readout = self.backbone.readout
        feat_dim = HIDDEN_FEATS[-1] * 2
        
        self.policy_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, action_size)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, g, feats):
        h = self.encoder(g, feats)
        graph_feat = self.readout(g, h)
        policy_logits = self.policy_head(graph_feat)
        value_pred = self.value_head(graph_feat)
        return policy_logits, value_pred

# --- 评估函数 (复用逻辑) ---
def evaluate_dataset(model, loader, criterion_mse, device):
    model.eval()
    total_loss_p = 0
    total_loss_v = 0
    count = 0
    
    all_p_true, all_p_pred = [], []
    all_v_true, all_v_pred = [], []
    
    with torch.no_grad():
        for batch_graph, batch_p, batch_v in loader:
            if batch_graph is None: continue
            
            policy_logits, value_pred = model(batch_graph, batch_graph.ndata['h'])
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            l_p = criterion_mse(policy_probs, batch_p)
            l_v = criterion_mse(value_pred.squeeze(), batch_v.squeeze())
            
            total_loss_p += l_p.item()
            total_loss_v += l_v.item()
            count += 1
            
            all_p_true.append(batch_p.cpu())
            all_p_pred.append(policy_probs.cpu())
            all_v_true.append(batch_v.cpu())
            all_v_pred.append(value_pred.cpu())
    
    if count == 0:
        return None
    
    avg_loss_p = total_loss_p / count
    avg_loss_v = total_loss_v / count
    
    # 合并数据
    p_true = torch.cat(all_p_true, dim=0).numpy()
    p_pred = torch.cat(all_p_pred, dim=0).numpy()
    v_true = torch.cat(all_v_true, dim=0).numpy()
    v_pred = torch.cat(all_v_pred, dim=0).numpy()
    
    # 计算 Metrics
    # 1. Policy Masked RMSE
    mask = (p_true > 1e-6).astype(float)
    diff = (p_pred - p_true) * mask
    p_masked_rmse = np.sqrt(np.sum(diff**2) / (np.sum(mask) + 1e-8))
    
    # 2. Policy Top-1 Accuracy
    top1_correct = 0
    total_samples = len(p_true)
    for i in range(total_samples):
        true_idxs = np.where(p_true[i] > 1e-6)[0]
        if len(true_idxs) == 0: continue
        pred_top1 = np.argmax(p_pred[i])
        if pred_top1 in true_idxs:
            top1_correct += 1
    top1_acc = top1_correct / total_samples
    
    # 3. Value Metrics
    v_rmse = np.sqrt(mean_squared_error(v_true, v_pred))
    v_mae = mean_absolute_error(v_true, v_pred)
    v_r2 = r2_score(v_true, v_pred)
    
    return {
        "loss_p": avg_loss_p,
        "loss_v": avg_loss_v,
        "p_masked_rmse": p_masked_rmse,
        "p_top1_acc": top1_acc,
        "v_rmse": v_rmse,
        "v_mae": v_mae,
        "v_r2": v_r2
    }

def print_metrics(name, metrics):
    if metrics is None:
        print(f"{name}: No data")
        return
    print(f"--- {name} ---")
    print(f"  Policy Loss: {metrics['loss_p']:.4f} | Masked RMSE: {metrics['p_masked_rmse']:.4f} | Top-1 Acc: {metrics['p_top1_acc']:.4f}")
    print(f"  Value Loss:  {metrics['loss_v']:.4f} | RMSE: {metrics['v_rmse']:.4f} | MAE: {metrics['v_mae']:.4f} | R²: {metrics['v_r2']:.4f}")

# --- 主流程 ---
def main():
    print("📂 Loading JSON data...")
    with open(DATA_PATH, 'r') as f:
        raw_data = json.load(f)
    
    total = len(raw_data)
    test_ratio = 1/12
    val_ratio_from_temp = (1/12) / (1 - test_ratio)
    
    indices = np.arange(total)
    np.random.seed(42)
    np.random.shuffle(indices)
    shuffled_data = [raw_data[i] for i in indices]
    
    n_test = int(total * test_ratio)
    data_temp, data_test = shuffled_data[:-n_test], shuffled_data[-n_test:]
    
    n_val = int(len(data_temp) * val_ratio_from_temp)
    data_train, data_val = data_temp[:-n_val], data_temp[-n_val:]
    
    print(f"\n📊 Data Split:")
    print(f"  Train:      {len(data_train):6d}")
    print(f"  Validation: {len(data_val):6d}")
    print(f"  Test:       {len(data_test):6d}")
    
    train_dataset = JointMoleculeDataset(data_train)
    val_dataset = JointMoleculeDataset(data_val)
    test_dataset = JointMoleculeDataset(data_test)
    
    if len(train_dataset) == 0:
        print("❌ No valid data!")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_joint, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_joint, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_joint, shuffle=False)
    
    node_feat_dim = train_dataset[0][0].ndata['h'].shape[1]
    model = JointGCNModel(node_feat_dim, ACTION_SPACE_SIZE).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\n🏋️ Start Joint Training (Policy + Value)...")
    start_time = time.time()
    
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        total_loss_p = 0
        total_loss_v = 0
        count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_graph, batch_p, batch_v in pbar:
            if batch_graph is None: continue
            
            optimizer.zero_grad()
            policy_logits, value_pred = model(batch_graph, batch_graph.ndata['h'])
            policy_probs = F.softmax(policy_logits, dim=-1)
            
            loss_p = criterion_mse(policy_probs, batch_p)
            loss_v = criterion_mse(value_pred.squeeze(), batch_v.squeeze())
            loss = LAMBDA_POLICY * loss_p + LAMBDA_VALUE * loss_v
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_p += loss_p.item()
            total_loss_v += loss_v.item()
            count += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss_p = total_loss_p / count
        avg_train_loss_v = total_loss_v / count
        
        # --- 验证集评估 ---
        val_metrics = evaluate_dataset(model, val_loader, criterion_mse, DEVICE)
        avg_val_loss = LAMBDA_POLICY * val_metrics['loss_p'] + LAMBDA_VALUE * val_metrics['loss_v']
        
        # --- 打印当前 Epoch 的 Train 和 Val 对比 ---
        print(f"\nEpoch {epoch+1}/{N_EPOCHS} Summary:")
        print(f"  [Train] Policy Loss: {avg_train_loss_p:.4f} | Value Loss: {avg_train_loss_v:.4f}")
        print(f"  [Val]   Policy Loss: {val_metrics['loss_p']:.4f} | Value Loss: {val_metrics['loss_v']:.4f} | Total: {avg_val_loss:.4f}")
        print(f"  [Val]   P_RMSE: {val_metrics['p_masked_rmse']:.4f} | V_RMSE: {val_metrics['v_rmse']:.4f} | V_R2: {val_metrics['v_r2']:.4f}")
        
        # 早停逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n⏱️ Training finished in {time.time() - start_time:.2f}s")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("✅ Loaded best model.")
    
    # --- 最终详细评估 (Train, Val, Test) ---
    print("\n" + "="*80)
    print("                  FINAL EVALUATION RESULTS (All Sets)                 ")
    print("="*80)
    
    # 重新评估三个集合
    train_metrics = evaluate_dataset(model, train_loader, criterion_mse, DEVICE)
    val_metrics = evaluate_dataset(model, val_loader, criterion_mse, DEVICE)
    test_metrics = evaluate_dataset(model, test_loader, criterion_mse, DEVICE)
    
    print("\n1. TRAINING SET PERFORMANCE:")
    print_metrics("Train", train_metrics)
    
    print("\n2. VALIDATION SET PERFORMANCE:")
    print_metrics("Validation", val_metrics)
    
    print("\n3. TEST SET PERFORMANCE (Final Report):")
    print_metrics("Test", test_metrics)
    
    print("\n" + "="*80)
    print("Summary Table (Test Set):")
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 45)
    if test_metrics:
        print(f"{'Policy Masked RMSE':<25} | {test_metrics['p_masked_rmse']:<15.4f}")
        print(f"{'Policy Top-1 Accuracy':<25} | {test_metrics['p_top1_acc']:<15.4f}")
        print(f"{'Value RMSE':<25} | {test_metrics['v_rmse']:<15.4f}")
        print(f"{'Value MAE':<25} | {test_metrics['v_mae']:<15.4f}")
        print(f"{'Value R²':<25} | {test_metrics['v_r2']:<15.4f}")
    print("="*80)
    print("🎉 Joint Training Complete!")

if __name__ == "__main__":
    main()