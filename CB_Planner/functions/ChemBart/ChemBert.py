import torch.nn as nn
import torch
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder
from CBTokenizer import CBTokenizer
import os
import torch.nn.functional as F
absdir = os.path.dirname(os.path.abspath(__file__))+"/"
class ChemBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BartConfig.from_pretrained(absdir + "config.json")
        self.shared = nn.Embedding(self.config.vocab_size,
                      self.config.d_model, self.config.pad_token_id)
        self.encoder = BartEncoder(self.config, self.shared)
        self.tokenizer = CBTokenizer()
        
    def forward(self, input_ids, attention_mask = None, 
                output_attentions = None, output_hidden_states = None):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True)
        return encoder_outputs
    def load(self):
        try:
            self.load_state_dict(torch.load(absdir + 'model/ChemBert.pth',map_location='cpu'))
        except Exception as err:
            print(err)
            print("new model")
    
    def load_from_bart(self):
        from ChemBart import ChemBart
        cb = ChemBart()
        self.shared.load_state_dict(cb.BartNN.model.shared.state_dict())
        self.encoder.load_state_dict(cb.BartNN.model.encoder.state_dict())
        del cb
    
class CB_END(nn.Module):
    '''
    this api uses the output of end token
    '''
    def __init__(self, out_type: int,
                 name: str, device: str = "cuda:0",
                 ran: int = 0):
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
        self.name = absdir + "model/"+name+'.pth'
        self.tokenizer = CBTokenizer()
        self.type = out_type
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BertNN = ChemBert()
        self.ran = ran
        if self.type == 1 or self.type == 2:
            self.linear = nn.Linear(1024, 1)
        elif self.type > 2:
            self.linear = nn.Linear(1024, self.type)
        else:
            raise("invalid type!")
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BertNN.load_from_bart()
            print("pre-trained model")
        else:
            print("new model")
            
    def forward(self, x):
        first_hidden = self.BertNN(input_ids=x, output_hidden_states=True).last_hidden_state[0][0]
        linear_out = self.linear(F.relu(first_hidden))
        if self.type == 1:
            if self.ran == 0:
                return linear_out[0]
            elif self.ran < 0:
                return torch.tanh(linear_out[0])*(-1)*self.ran
            else:
                return torch.sigmoid(linear_out[0])*self.ran
                #ref: (0,10)
        elif self.type == 2:
            return torch.sigmoid(linear_out[0])
        else:
            return torch.sigmoid(linear_out)
    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        '''
        data: (one piece of input as smiles string, label)
        label: for regression/ bi-classification, float; for multi-classification, one-hot
        '''
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        if self.type == 1:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.BCELoss()
        bestval = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            ep_loss = 0.0
            cor = 0.0
            count = 0
            self.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count += 1
                out = self(inp.to(self.device))
                #print(out,i[1][0],flush = True)
                cor += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                label = torch.tensor(i[1]).to(self.device)
                loss = criterion(out,label)
                ep_loss = ep_loss + loss.item()
                loss.backward()
                optimizer.step()
            cor = self._post_proc(cor,count)
            print("epoch loss:{}, train_acc:{},train_count:{}".format(ep_loss,cor,count))
            self.eval()
            corval = 0.0
            count = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    inp = self.tokenizer.encoder(i[0])
                    if len(inp) == 0:
                        continue
                    count += 1
                    out = self(inp.to(self.device))
                    corval += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
            corval = self._post_proc(corval,count)
            print("validation_acc:",corval,",val_count:",count,flush=True)
            if (bestval is None) or\
                    (self.type == 1 and corval < bestval) or\
                    (self.type > 1 and corval > bestval):
                bestval = corval
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
    def test(self, test_data, return_detail = False):
        acc = 0.0
        self.eval()
        self.to(self.device)
        ans = []
        count = 0
        with torch.no_grad():
            for i in test_data:
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count+=1
                out = self(inp.to(self.device))
                acc += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                if return_detail:
                    ans.append([i[1],out])
            acc = self._post_proc(acc, count)
            print("test_acc:", acc, flush=True)
        return (acc, ans)
    def _get_acc(self,out,label) -> float:
        if self.type == 2:
            if (out<0.5 and label<0.5) or (out>=0.5 and label>=0.5):
                return 1.0
            else:
                return 0.0
        elif self.type == 1:
            return (out - label)**2
        else:
            return float(torch.argmax(out) == torch.argmax(label))
    def _post_proc(self,acc:float,num:int) -> float:
        acc = acc/num
        if self.type == 1:
            acc = acc**0.5
            #rmse
        return acc
    def ret_x_y_list(self,data):
        assert self.type == 1, "only for regression use"
        ans = []
        with torch.no_grad():
            for i in data:
                out = self(self.tokenizer.encoder(i[0]).to(self.device))
                ans.append([i[1],out])
        return ans

class CB_mul_END(nn.Module):
    '''
    this api uses the output of multi tokens at the end
    '''
    def __init__(self, name: str, device: str = "cuda:0"):
        super().__init__()
        self.name = absdir + "model/"+name+'.pth'
        self.tokenizer = CBTokenizer()
        self.config=BartConfig.from_pretrained("config.json")
        self.BertNN = ChemBert()
        self.linear1 = nn.Linear(1024, 1)
        self.linear2 = nn.Linear(1024, 1)
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BertNN.load_from_bart()
            print("pre-trained model")
        else:
            print("new model")
            
    def forward(self, x):
        hidden1, hidden2 = self.BertNN(input_ids=x).last_hidden_state[0][0:2]
        linear_out1 = self.linear1(F.relu(hidden1)) #temperature
        linear_out2 = 150.0 * torch.sigmoid(self.linear2(F.relu(hidden2))) #yield
        return torch.cat((linear_out1, linear_out2))
    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        '''
        data: (one piece of input as smiles string, label)
        label: for regression/ bi-classification, float; for multi-classification, one-hot
        '''
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = torch.nn.MSELoss()
        bestval_temp = None
        bestval_yiel = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            ep_loss = 0.0
            cor_temp = 0.0
            cor_yiel = 0.0
            count_temp = 0
            count_yiel = 0
            self.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                if type(i[1][0]) != type(1.1) and type(i[1][1]) != type(1.1):
                    continue
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                out = self(inp.to(self.device))
                out_no_grad = out.tolist()
                #print(out,i[1][0],flush = True)
                if type(i[1][0]) == type(1.1) and type(i[1][1]) == type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_temp += 1
                    count_yiel += 1
                    label = torch.tensor(i[1]).to(self.device)
                    loss = criterion(out,label)
                elif type(i[1][0]) != type(1.1):
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_yiel += 1
                    label = torch.tensor(i[1][1]).to(self.device)
                    loss = criterion(out[1], label)
                elif type(i[1][1]) != type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    count_temp += 1
                    label = torch.tensor(i[1][0]).to(self.device)
                    loss = criterion(out[0], label)
                ep_loss = ep_loss + loss.item()
                loss.backward()
                optimizer.step()
            cor_temp = (cor_temp/count_temp)**0.5
            cor_yiel = (cor_yiel/count_yiel)**0.5
            print("epoch loss:{}\n train_temp_acc:{},train_temp_count:{}"
                  .format(ep_loss,cor_temp,count_temp))
            print("train_yiel_acc:{},train_yiel_count:{}"
                  .format(cor_yiel,count_yiel))
            self.eval()
            cor_temp = 0.0
            cor_yiel = 0.0
            count_temp = 0
            count_yiel = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    if type(i[1][0]) != type(1.1) and type(i[1][1]) != type(1.1):
                        continue
                    inp = self.tokenizer.encoder(i[0])
                    if len(inp) == 0:
                        continue
                    out = self(inp.to(self.device))
                    out_no_grad = out.tolist()
                    if type(i[1][0]) == type(1.1) and type(i[1][1]) == type(1.1):
                        cor_temp += (i[1][0] - out_no_grad[0])**2
                        cor_yiel += (i[1][1] - out_no_grad[1])**2
                        count_temp += 1
                        count_yiel += 1
                    elif type(i[1][0]) != type(1.1):
                        cor_yiel += (i[1][1] - out_no_grad[1])**2
                        count_yiel += 1
                    elif type(i[1][1]) != type(1.1):
                        cor_temp += (i[1][0] - out_no_grad[0])**2
                        count_temp += 1
            cor_temp = (cor_temp/count_temp)**0.5
            cor_yiel = (cor_yiel/count_yiel)**0.5
            print("epoch loss:{}\n val_temp_acc:{},val_temp_count:{}"
                  .format(ep_loss,cor_temp,count_temp))
            print("val_yiel_acc:{},val_yiel_count:{}"
                  .format(cor_yiel,count_yiel))
            if (bestval_temp is None) or\
                (cor_temp < bestval_temp and cor_yiel < bestval_yiel):
                bestval_temp = cor_temp
                bestval_yiel = cor_yiel
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
    def test(self, test_data, return_detail = False):
        cor_temp = 0.0
        cor_yiel = 0.0
        count_temp = 0
        count_yiel = 0
        self.eval()
        self.to(self.device)
        ans = []
        count = 0
        with torch.no_grad():
            for i in test_data:
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count+=1
                out = self(inp.to(self.device))
                out_no_grad = out.tolist()
                if type(i[1][0]) == type(1.1) and type(i[1][1]) == type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_temp += 1
                    count_yiel += 1
                elif type(i[1][0]) != type(1.1):
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_yiel += 1
                elif type(i[1][1]) != type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    count_temp += 1
                if return_detail:
                    ans.append([i[1],out_no_grad])
            cor_temp = (cor_temp/count_temp)**0.5
            cor_yiel = (cor_yiel/count_yiel)**0.5
            print("test_temp_acc:", cor_temp, "test_yiel_acc:", cor_yiel, flush=True)
        return ((cor_temp, cor_yiel), ans)
    
        
