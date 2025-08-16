import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BartConfig, BartForConditionalGeneration
from CBTokenizer import CBTokenizer
import os
import pickle
import matplotlib.pyplot as plt
from rdkit import Chem
import argparse
import sys
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, classification_report


# ================== dataset ==================
class CBData(Dataset):
    def __init__(self, data, maxlen=1024):
        self.data = data
        self.maxlen = maxlen

    def __getitem__(self, index):
        encoded = self.data[index]
        
        return encoded[0], encoded[1], encoded[2]

    def __len__(self):
        return len(self.data)

# ================== core model ==================


class MolProperty(nn.Module):
    def __init__(self, out_type: int,
                 name: str, pre_model: str, device: str = "cuda:0",
                 ran: int = 0, epoch_stop: int=20):
        '''
        out_type:
        1: regression
            ran: if ran<0, range in [-ran,ran]
                    if ran>0, range in [0,ran]
                    if ran = 0, range in R
        2: binary classification
        n>=3: ont-hot-encoding classification with n classes
        '''
        super().__init__()
        self.name = name+'.pth'
        self.save_path = os.path.join('checkpoints_test', self.name)
        self.pre_model = os.path.join("ChemBart_model", pre_model + ".pth")
        self.tokenizer = CBTokenizer()
        self.type = out_type
        self.epoch_stop = epoch_stop
        self.ran = ran

        # load Bart model
        self.config=BartConfig.from_pretrained("config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        hidden_size = self.config.d_model
        
        
        # 输出层
        if self.type == 1 or self.type == 2:
            self.linear = nn.Linear(hidden_size, 1)
        elif self.type > 2:
            self.linear = nn.Linear(hidden_size, self.type)
        else:
            raise ValueError("Invalid output type.")

        self.device = torch.device(device)
        self._load_weights()
    
    def _load_weights(self):
        """load checkpoints"""
        if os.path.exists(self.save_path):
            self.load_state_dict(torch.load(self.save_path,map_location='cpu'))
            print(f"Fine-tuned model loaded from {self.save_path}")
        elif os.path.exists(self.pre_model):
            #bart_state = torch.load(self.pre_model, map_location='cpu')
            self.BartNN.load_state_dict(torch.load(self.pre_model,map_location='cpu'))
            print(f"Pre-trained model loaded from {self.pre_model}")
        else:
            print("No checkpoint found. Using random initialization.")
            
    def forward(self, input_ids, attention_mask=None):
        """前向传播：仅使用 encoder 的 [EOS] token 表示"""
        #encoder_output = self.bart_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        #cls_output = encoder_output[:, -1, :]  # 取最后一个 token 
        last_hidden = self.BartNN(input_ids=input_ids, decoder_input_ids=input_ids,
                                  attention_mask=attention_mask, decoder_attention_mask=attention_mask,
                                  return_dict=True, output_hidden_states=True).decoder_hidden_states[-1][:, -1, :]
        linear_out = self.linear(torch.sigmoid(last_hidden))
        
        return linear_out.squeeze(1)

    def _get_acc(self, outputs, labels):
        """向量化准确率计算"""
        if self.type == 1:
            return (torch.abs(outputs - labels) < 0.5).float().sum().item()
        elif self.type == 2:
            preds = (outputs.sigmoid() > 0.5).float()
            return (preds == labels).float().sum().item()
        else:
            preds = outputs.argmax(dim=1)
            true = labels.argmax(dim=1)
            return (preds == true).float().sum().item()

    def _post_proc(self, cor, count):
        return cor / count if count > 0 else 0


    def batchtrain2(self, traindata: list, valdata, epoch: int, batch_size: int = 32):
        """更改原数据切片操作"""

        self.to(self.device)

        optimizer = optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = self._get_criterion()

        train_dataset = CBData(traindata)
        val_dataset = CBData(valdata)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_metric = None
        no_improvement_count = 0

        for i in range(epoch):
            print(f"Epoch {i}", flush=True)
            self.train()
            ep_loss = 0.0
            cor_train = 0.0
            count_train = 0

            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.float().to(self.device)

                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                ep_loss += loss.item()
                cor_train += self._get_acc(outputs, labels)
                count_train += input_ids.shape[0]

            print(f"Train Loss: {ep_loss}, Train Acc: {cor_train / count_train}")

            self.eval()
            cor_val = 0.0
            count_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.float().to(self.device)

                    outputs = self(input_ids, attention_mask)
                    cor_val += self._get_acc(outputs, labels)
                    count_val += input_ids.shape[0]

            val_acc = cor_val / count_val
            print(f"Validation Acc: {val_acc}, Val Count: {count_val}", flush=True)

            # Early stopping
            is_better = False
            if self.type == 1:
                is_better = best_val_metric is None or val_acc < best_val_metric
            else:
                is_better = best_val_metric is None or val_acc > best_val_metric

            if is_better:
                best_val_metric = val_acc
                torch.save(self.state_dict(), self.save_path)
                print("Model saved!", flush=True)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.epoch_stop:
                print(f"No improvement in the last {self.epoch_stop} epochs. Training stopped.")
                break

    def test(self, test_data, batch_size=32):
        """测试流程并输出 ROC 曲线"""
        self.eval()
        self.to(self.device)
        y_true, y_scores = [], []

        testset = CBData(test_data)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self(input_ids, attention_mask).sigmoid().cpu().numpy()
                labels = labels.cpu().numpy()

                y_true.extend(labels.tolist())
                y_scores.extend(outputs.tolist())

        # Accuracy
        if self.type == 2:
            y_pred = [1 if score >= 0.5 else 0 for score in y_scores]
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_scores)
            fpr, tpr, _ = roc_curve(y_true, y_scores)

            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()

            print(f"Test Accuracy: {acc}, AUC: {auc}")
            return acc, auc
        else:
            y_pred = [score.index(max(score)) for score in y_scores]
            y_true_idx = [label.index(1) if isinstance(label, list) else int(label) for label in y_true]
            acc = accuracy_score(y_true_idx, y_pred)
            print(f"Test Accuracy: {acc}")
            return acc

        
    def transform(self, smiles_list):

        self.to(self.device)
        outputs = []

        with torch.no_grad():
            for smile in smiles_list:
                # 将 SMILES 编码为输入张量
                inp = self.tokenizer.encoder(smile)
                out = inp.to(self.device)

                # 获取输出向量
                out = self.BartNN(input_ids=out, decoder_input_ids=out, return_dict=True, output_hidden_states=True).decoder_hidden_states[-1][0][-1]

                # 将输出向量添加到结果列表中
                outputs.append(out.cpu().numpy())

        return outputs

    def _get_criterion(self):
        """根据任务选择合适的损失函数"""
        if self.type == 1:
            return nn.MSELoss()
        elif self.type == 2:
            return nn.BCEWithLogitsLoss()
           
        else:
            return nn.CrossEntropyLoss()
        



class MolProperty_encoder_only(nn.Module):
    def __init__(self, out_type: int,
                 name: str, pre_model: str, device: str = "cuda:0",
                 ran: int = 0, epoch_stop: int=20):
        '''
        out_type:
        1: regression
            ran: if ran<0, range in [-ran,ran]
                    if ran>0, range in [0,ran]
                    if ran = 0, range in R
        2: binary classification
        n>=3: ont-hot-encoding classification with n classes
        '''
        super().__init__()
        self.name = name+'.pth'
        self.save_path = os.path.join('checkpoints_test', self.name)
        self.pre_model = os.path.join("ChemBart_model", pre_model + ".pth")
        self.tokenizer = CBTokenizer()
        self.type = out_type
        self.epoch_stop = epoch_stop
        self.ran = ran

        # load Bart model
        self.config=BartConfig.from_pretrained("config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.bart_encoder = self.BartNN.get_encoder()
        hidden_size = self.config.d_model
        
        
        # 输出层
        if self.type == 1 or self.type == 2:
            self.linear = nn.Linear(hidden_size, 1)
        elif self.type > 2:
            self.linear = nn.Linear(hidden_size, self.type)
        else:
            raise ValueError("Invalid output type.")

        self.device = torch.device(device)
        self._load_weights()
    
    def _load_weights(self):
        """load checkpoints"""
        if os.path.exists(self.save_path):
            self.load_state_dict(torch.load(self.save_path,map_location='cpu'))
            print(f"Fine-tuned model loaded from {self.save_path}")
        elif os.path.exists(self.pre_model):
            #bart_state = torch.load(self.pre_model, map_location='cpu')
            self.BartNN.load_state_dict(torch.load(self.pre_model,map_location='cpu'))
            self.bart_encoder = self.BartNN.get_encoder()
            print(f"Pre-trained model loaded from {self.pre_model}")
        else:
            print("No checkpoint found. Using random initialization.")
            
    def forward(self, input_ids, attention_mask=None):
        #encoder_output = self.bart_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        #cls_output = encoder_output[:, -1, :]  # 取最后一个 token 
        encoder_output = self.bart_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_output = encoder_output[:, 0, :]  # 取第一个 token ([CLS])
        linear_out = self.linear(torch.sigmoid(cls_output))
        return linear_out.squeeze(1)

    def _get_acc(self, outputs, labels):
        """向量化准确率计算"""
        if self.type == 1:
            return (torch.abs(outputs - labels) < 0.5).float().sum().item()
        elif self.type == 2:
            preds = (outputs.sigmoid() > 0.5).float()
            return (preds == labels).float().sum().item()
        else:
            preds = outputs.argmax(dim=1)
            true = labels.argmax(dim=1)
            return (preds == true).float().sum().item()

    def _post_proc(self, cor, count):
        return cor / count if count > 0 else 0

    def batchtrain(self, traindata: list, valdata, epoch: int, batch_size: int = 32):

        self.to(self.device)

        optimizer = optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = self._get_criterion()

        train_dataset = CBData(traindata)
        val_dataset = CBData(valdata)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_metric = None
        no_improvement_count = 0

        for i in range(epoch):
            #print(f"Epoch {i}", flush=True)
            self.train()
            ep_loss = 0.0
            cor_train = 0.0
            count_train = 0

            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.float().to(self.device)

                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                ep_loss += loss.item()
                cor_train += self._get_acc(outputs, labels)
                count_train += input_ids.shape[0]

            #print(f"Train Loss: {ep_loss}, Train Acc: {cor_train / count_train}")

            self.eval()
            cor_val = 0.0
            count_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.float().to(self.device)

                    outputs = self(input_ids, attention_mask)
                    cor_val += self._get_acc(outputs, labels)
                    count_val += input_ids.shape[0]

            val_acc = cor_val / count_val
            #print(f"Validation Acc: {val_acc}, Val Count: {count_val}", flush=True)

            # Early stopping
            is_better = False
            if self.type == 1:
                is_better = best_val_metric is None or val_acc < best_val_metric
            else:
                is_better = best_val_metric is None or val_acc > best_val_metric

            if is_better:
                best_val_metric = val_acc
                torch.save(self.state_dict(), self.save_path)
                print("Model saved!", flush=True)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.epoch_stop:
                print(f"No improvement in the last {self.epoch_stop} epochs. Training stopped.")
                break

    def test(self, test_data, batch_size=32):
        """测试流程并输出 ROC 曲线"""
        self.eval()
        self.to(self.device)
        y_true, y_scores = [], []

        testset = CBData(test_data)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self(input_ids, attention_mask).sigmoid().cpu().numpy()
                labels = labels.cpu().numpy()

                y_true.extend(labels.tolist())
                y_scores.extend(outputs.tolist())

        # Accuracy
        if self.type == 2:
            y_pred = [1 if score >= 0.5 else 0 for score in y_scores]
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_scores)
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            '''
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()
            '''
            #print(f"Test Accuracy: {acc}, AUC: {auc}")
            return acc, auc
        else:
            y_pred = [score.index(max(score)) for score in y_scores]
            y_true_idx = [label.index(1) if isinstance(label, list) else int(label) for label in y_true]
            acc = accuracy_score(y_true_idx, y_pred)
            print(f"Test Accuracy: {acc}")
            return acc

    def transform(self, smiles_list):
        """将 SMILES 编码为特征向量"""
        self.to(self.device)
        outputs = []
        with torch.no_grad():
            for smile in smiles_list:
                input_ids = self.tokenizer.encoder(smile)
                input_ids = input_ids.to(self.device)
                output = self.bart_encoder(input_ids=input_ids, return_dict=True).last_hidden_state
                outputs.append(output[:, -1, :].cpu().numpy())
        return outputs

    def _get_criterion(self):
        """根据任务选择合适的损失函数"""
        if self.type == 1:
            return nn.MSELoss()
        elif self.type == 2:
            return nn.BCEWithLogitsLoss()
            #return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()