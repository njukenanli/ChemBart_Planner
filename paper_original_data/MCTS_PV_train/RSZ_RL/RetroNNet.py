# RetroNNet.py
import sys
sys.path.append('..')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
class RetroNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.action_size = output_dim
        super(RetroNNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=0.3, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=0.3, training=self.training)
        pi = self.fc3(s)
        v = self.fc4(s)
        return torch.softmax(pi, dim=1), torch.tanh(v)

    def fit(self, data, tr, val, te, epoch=200, batch_size=64, dev="cuda:0"):

        device = torch.device(dev)
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)
        criterion = torch.nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        bestv, bestp = float('inf'), float('inf')

        with torch.no_grad():
            all_fea = torch.stack([x[0] for x in data]).to(device)
            all_labelp = torch.stack([x[1][0] for x in data]).to(device, dtype=torch.float32)
            all_labelv = torch.stack([x[1][1] for x in data]).to(device, dtype=torch.float32)

        # 创建数据划分索引
        train_indices = torch.arange(tr)
        val_indices = torch.arange(tr, tr+val)
        test_indices = torch.arange(tr+val, tr+val+te)

        for e in range(epoch):
            self.train()
            print(f"epoch {e}", flush=True)
            
            # 打乱训练集索引
            train_indices = torch.randperm(tr)
            
            total_loss_v = 0.0
            total_loss_p = 0.0
            
            for idx in range(0, tr, batch_size):
                # 获取批次数据
                indices = train_indices[idx:idx+batch_size]
                fea = all_fea[indices]
                labelp = all_labelp[indices]
                labelv = all_labelv[indices]

                # 混合精度训练
                with torch.cuda.amp.autocast():
                    p, v = self(fea)
                    lp = criterion(p, labelp)
                    lv = criterion(v, labelv)
                    loss = lp + lv

                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 后处理需要float32精度
                with torch.no_grad():
                    p = p.float()  # 转换为float32
                    # 应用掩码
                    mask = (labelp >= 1e-8).float()
                    p = p * mask
                    p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)

                # 累积损失
                total_loss_v += lv.item() * len(fea)
                total_loss_p += self.RMSE(p, labelp).item() * len(fea)

            # 打印训练损失
            avg_loss_v = total_loss_v / tr
            avg_loss_p = total_loss_p / tr
            print(f"Train Mean RMSE v:{avg_loss_v:.4f}, p:{avg_loss_p:.4f}")

            # 验证阶段
            val_loss_v, val_loss_p = self._evaluate(
                all_fea[val_indices],
                all_labelp[val_indices],
                all_labelv[val_indices],
                batch_size
            )
            print(f"Val Mean RMSE v:{val_loss_v:.4f}, p:{val_loss_p:.4f}")

            # 测试阶段
            test_loss_v, test_loss_p = self._evaluate(
                all_fea[test_indices],
                all_labelp[test_indices],
                all_labelv[test_indices],
                batch_size
            )
            print(f"Test Mean RMSE v:{test_loss_v:.4f}, p:{test_loss_p:.4f}")
    
    def test(self, data, batch_size=64, dev="cuda:0"):
        device = torch.device(dev)
        self.to(device)
        
        # 转换为张量
        with torch.no_grad():
            fea = torch.stack([x[0] for x in data]).to(device)
            labelp = torch.stack([x[1][0] for x in data]).to(device, dtype=torch.float32)
            labelv = torch.stack([x[1][1] for x in data]).to(device, dtype=torch.float32)
        
        return self._evaluate(fea, labelp, labelv, batch_size)

    def _evaluate(self, fea, labelp, labelv, batch_size):
        """通用评估函数"""
        self.eval()
        total_loss_v = 0.0
        total_loss_p = 0.0
        dataset_size = fea.size(0)

        with torch.no_grad():
            for idx in range(0, dataset_size, batch_size):
                # 获取批次数据
                batch_fea = fea[idx:idx+batch_size]
                batch_labelp = labelp[idx:idx+batch_size]
                batch_labelv = labelv[idx:idx+batch_size]

                # 混合精度推理
                with torch.cuda.amp.autocast():
                    p, v = self(batch_fea)
                    loss_v = F.mse_loss(v.float(), batch_labelv.float())

                # 后处理需要float32精度
                p = p.float()
                # 应用掩码并归一化
                mask = (batch_labelp >= 1e-8).float()
                p = p * mask
                p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)
                
                # 计算RMSE
                loss_p = self.RMSE(p, batch_labelp)

                # 累积损失
                total_loss_v += loss_v.item() * batch_fea.size(0)
                total_loss_p += loss_p.item() * batch_fea.size(0)

        return total_loss_v / dataset_size, total_loss_p / dataset_size
    
    def RMSE(self, p, labelp, eps=1e-8):
        mask = (labelp >= 1e-8).float()
        valid_count = mask.sum(dim=1).clamp(min=eps)  # 防止除零
        diff = (p - labelp) * mask
        mse = (diff.pow(2).sum(dim=1) / valid_count)
        return mse.sqrt().mean()