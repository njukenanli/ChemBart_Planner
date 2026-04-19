import json
import torch
import os
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BartForConditionalGeneration, BartConfig
from typing import List, Tuple, Dict

class OptimizedCBTokenizer:
    def __init__(self, vocab_path='vocab.json'):
        self.vocab = self._load_vocab(vocab_path)
        self.vocab_keys = list(self.vocab.keys())
        self.d1 = {k: v for k, v in self.vocab.items() if len(k) == 1}
        self.d2 = {k: v for k, v in self.vocab.items() if len(k) == 2}
        self.d5 = {k: v for k, v in self.vocab.items() if len(k) == 5}
        self.encoding_cache = {}
        self.max_len = 0

    def _load_vocab(self, path):
        try:
            with open(path) as f:
                return json.load(f)
        except:
            raise FileNotFoundError(f"Vocabulary file {path} not found")

    def preprocess_dataset(self, dataset: List) -> List:
        """统一编码并计算最大长度"""
        processed_data = []
        max_len = 0
        
        for item in dataset:
            if isinstance(item[0], (list, tuple)) and len(item[0]) == 2:  # MCTS格式
                smi, candidates = item[0]
                encoded_smi = self._encode_smiles(smi)
                encoded_cands = [self._encode_smiles(c) for c in candidates]
                
                if encoded_smi and all(encoded_cands):
                    current_max = max(len(encoded_smi), max(len(c) for c in encoded_cands))
                    max_len = max(max_len, current_max)
                    processed_data.append((encoded_smi, encoded_cands, item[1]))
            else:  # 普通格式
                encoded = self._encode_smiles(item[0])
                if encoded:
                    max_len = max(max_len, len(encoded))
                    processed_data.append((encoded, [], item[1]))  # 空候选列表
        
        self.max_len = max_len + 2  # 统一长度
        return processed_data

    def _encode_smiles(self, smi: str) -> List[int]:
        """编码单个SMILES字符串"""
        if smi in self.encoding_cache:
            return self.encoding_cache[smi]
            
        tokens = []
        i = 0
        while i < len(smi):
            if smi[i] == " ":
                i += 1
                continue
                
            # 优先匹配长token
            for token_len in [5, 2, 1]:
                if i <= len(smi)-token_len and smi[i:i+token_len] in getattr(self, f'd{token_len}'):
                    tokens.append(getattr(self, f'd{token_len}')[smi[i:i+token_len]])
                    i += token_len
                    break
            else:
                print(f"Unknown token in SMILES: {smi[i:]}")
                return None
                
        self.encoding_cache[smi] = tokens
        return tokens

    def batch_encode(self, smiles_list: List[str]) -> torch.Tensor:
        """批量编码并填充"""
        encoded = []
        for smi in smiles_list:
            tokens = self._encode_smiles(smi)
            if tokens:
                padded = tokens + [self.vocab["<pad>"]] * (self.max_len - len(tokens))
                encoded.append(torch.tensor(padded[:self.max_len]))
        return pad_sequence(encoded, batch_first=True, padding_value=self.vocab["<pad>"]) if encoded else torch.empty(0, self.max_len)

class BatchCB_END(nn.Module):
    def __init__(self, out_type: int, name: str, premodel: str, device: str = "cuda:0", ran: int = 0):
        super().__init__()
        self.config = BartConfig.from_pretrained("config.json")
        self.BartNN = BartForConditionalGeneration(self.config)
        self.type = out_type
        self.ran = ran
        self.linear = nn.Linear(1024, 1 if out_type in [1, 2] else out_type)
        self._load_weights(name, premodel)
        self.device = torch.device(device)

    def _load_weights(self, name, premodel):
        if os.path.exists(name):
            self.load_state_dict(torch.load(name, map_location='cpu'))
        elif os.path.exists(premodel):
            self.BartNN.load_state_dict(torch.load(premodel, map_location='cpu'))
        else:
            print("Initialized new model")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.BartNN(input_ids=x, decoder_input_ids=x, return_dict=True, output_hidden_states=True)
        last_hidden = outputs.decoder_hidden_states[-1][:, -1, :]
        linear_out = self.linear(F.relu(last_hidden))
        
        if self.type == 1:
            return linear_out.squeeze(-1) if self.ran == 0 else \
                   torch.tanh(linear_out).squeeze(-1) * (-self.ran if self.ran < 0 else self.ran)
        elif self.type == 2:
            return torch.sigmoid(linear_out).squeeze(-1)
        return torch.softmax(linear_out, dim=-1)

class BatchCB_MCTS:
    def __init__(self, savename: str, premodel: str, dev: str = "cuda:0"):
        self.device = torch.device(dev)
        self.core = BatchCB_END(1, savename, premodel, dev)
        self.tokenizer = OptimizedCBTokenizer()
        self.pad_token = self.tokenizer.vocab["<pad>"]

        # 确保模型在指定设备上
        self.core.to(self.device)
    
    def _prepare_batch(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """统一处理设备转移"""
        # 输入数据
        smi_tensors = [torch.tensor(x[0], device=self.device) for x in batch]
        smi_batch = pad_sequence(smi_tensors, batch_first=True, padding_value=self.pad_token)
        # 候选数据
        all_cands = [c for x in batch for c in x[1]]
        cand_batch = pad_sequence(
            [torch.tensor(c, device=self.device) for c in all_cands],
            batch_first=True, padding_value=self.pad_token
        ) if all_cands else None
        
        # 标签数据（自动转移到模型设备）
        v_labels = torch.tensor([x[2][0] for x in batch], device=self.device, dtype=torch.float)
        p_labels = torch.cat([torch.tensor(x[2][1], device=self.device) for x in batch if x[2][1]]) \
                  if any(x[2][1] for x in batch) else None
        
        return smi_batch, cand_batch, v_labels, p_labels
        
    def forward(self, smi_batch: torch.Tensor, cand_batch: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量处理tensor输入"""
        v_pred = torch.tanh(self.core(smi_batch))
        p_pred = torch.softmax(self.core(cand_batch), dim=0) if cand_batch is not None else torch.empty(0, device=self.device)
        return v_pred, p_pred

    def batch_train(self, data: List, epochs: int, tr: int, val: int, te: int, batch_size: int = 32):
        processed_data = self.tokenizer.preprocess_dataset(data)
        train_data, val_data, test_data = processed_data[:tr], processed_data[tr:tr+val], processed_data[tr+val:]
        
        optimizer = torch.optim.AdamW(self.core.parameters(), lr=1e-6, weight_decay=1e-6)
        best_val = float('inf')
        
        for epoch in range(epochs):
            self._run_epoch(epoch, train_data, val_data, test_data, batch_size, optimizer, best_val)

    def _run_epoch(self, epoch, train_data, val_data, test_data, batch_size, optimizer, best_val):
        print(f"\nepoch {epoch}")
        
        # 训练阶段
        self.core.train()
        train_v, train_p, train_cnt = 0.0, 0.0, 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            if not batch: continue
            
            # 准备批量数据（自动处理设备）
            smi_batch, cand_batch, v_labels, p_labels = self._prepare_batch(batch)
            
            # 前向传播
            optimizer.zero_grad()
            v_pred, p_pred = self.forward(smi_batch, cand_batch)
            
            # 计算损失
            loss_v = F.mse_loss(v_pred, v_labels)
            loss_p = F.mse_loss(p_pred, p_labels) if p_labels is not None else 0
            total_loss = loss_v + loss_p
            total_loss.backward()
            optimizer.step()
            
            # 记录指标
            batch_size = len(batch)
            train_v += loss_v.sqrt().item() * batch_size
            train_p += loss_p.sqrt().item() * batch_size if p_labels is not None else 0
            train_cnt += batch_size

        # 打印训练结果
        print(f"train_v_rmse:{train_v/train_cnt:.6f}, train_p_rmse:{train_p/train_cnt:.6f}, train_count:{train_cnt}")
        
        # 验证和测试
        val_v, val_p, val_cnt = self._evaluate(val_data)
        test_v, test_p, test_cnt = self._evaluate(test_data)
        
        print(f"val_v_rmse:{val_v:.6f}, val_p_rmse:{val_p:.6f}, val_count:{val_cnt}")
        print(f"test_v_rmse:{test_v:.6f}, test_p_rmse:{test_p:.6f}, test_count:{test_cnt}")
        
        # 保存最佳模型
        if val_v < best_val:
            torch.save(self.core.state_dict(), self.core.name)
            print(f"model refreshed with val_v_rmse:{val_v:.6f}")

    def _evaluate(self, data: List) -> Tuple[float, float, int]:
        self.core.eval()
        v_rmse, p_rmse, count = 0.0, 0.0, 0
        
        with torch.no_grad():
            for encoded_smi, encoded_cands, (v_label, p_label) in data:
                # 处理输入
                smi_tensor = torch.tensor([encoded_smi], device=self.device)
                
                # 处理候选
                if encoded_cands:
                    cand_batch = pad_sequence(
                        [torch.tensor(c, device=self.device) for c in encoded_cands],
                        batch_first=True, padding_value=self.pad_token
                    )
                    v_pred, p_pred = self.forward(smi_tensor, cand_batch)
                    
                    # 标签数据转移到相同设备
                    p_label_tensor = torch.tensor(p_label, device=self.device)
                    v_rmse += (v_pred.item() - v_label) ** 2
                    p_rmse += ((p_pred - p_label_tensor) ** 2).mean().sqrt().item()
                    count += 1
        
        return (
            (v_rmse / count) ** 0.5 if count > 0 else 0.0,
            p_rmse / count if count > 0 else 0.0,
            count
        )