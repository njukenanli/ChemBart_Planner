#Configuration
TRAIN = True
USE_LSTM = False
TEMPLATE_NUM = 15
DEVICE = "cpu" #"cuda:0"
EPOCH = 50
TOPK = 10

import json
from ChemBart import CB_END, CB_LSTM
import copy
import torch

class CBTemplateCls():
    def __init__(self, LSTM, template_num, device):
        if LSTM:
            self.model = CB_LSTM(out_type = template_num, name = "CB4TemplateClsLSTM", device = device)
        else:
            self.model = CB_END(out_type = template_num, name = "CB4TemplateCls", device = device)
        self.tokenizer = self.model.tokenizer.encoder
        self.template_num = template_num
    def pred(self, product_smiles):
        '''
        input smiles
        output probability distribution of templates
        '''
        inp = self.tokenizer("<msk>>>" + product_smiles)
        if len(inp)==0:
            print(product_smiles, ": invalid input, continue!")
            return []
        return self.model(inp.to(self.model.device))
    def topk_idx(self, prob_distribution, k = 10):
        '''
        input probability distribution of templates: list
        output top-k choice of template indexes
        '''
        dist = torch.tensor(prob_distribution)
        toplist = [-1]*k
        for i in range(k):
            toplist[i] = torch.argmax(dist).item()
            dist[toplist[i]] = 0.0
        return toplist
    def fit(self, train, val, epoch, k = 10):
        self.model.to(self.model.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = torch.nn.BCELoss()
        besttopk = None
        for e in range(epoch):
            print("epoch", e, flush = True)
            ep_loss = 0.0
            topk = [0]*k
            self.model.train()
            for i in train:
                optimizer.zero_grad()
                out = self.pred(i[0])
                if len(out) == 0:
                    continue
                label = [0.0]*self.template_num
                label[i[1]] = 1.0
                label = torch.tensor(label).to(self.model.device)
                loss = criterion(out,label)
                ep_loss = ep_loss + loss.item()
                loss.backward()
                optimizer.step()
                piece_choice = self.topk_idx(out.tolist(), k = k)
                for j in range(k):
                    if piece_choice[j] == i[1]:
                        topk[j] += 1
                        break
            for j in range(1,k):
                topk[j] += topk[j-1]
            for j in range(k):
                topk[j] /= len(train)
            print("train loss:{}, topk:{}".format(ep_loss, topk))
            topk = self.test(val, k)
            print("validation topk:{}".format(topk))
            if (besttopk is None) or topk[k-1] > besttopk:
                besttopk = topk[k-1]
                torch.save(self.model.state_dict(), self.model.name)
                print("model refreshed!", flush = True)

    def test(self, test, k=10):
        self.model.eval()
        self.model.to(self.model.device)
        topk = [0]*k
        with torch.no_grad():
            for i in test:
                out = self.pred(i[0])
                if len(out) == 0:
                    continue
                piece_choice = self.topk_idx(out.tolist(), k = k)
                for j in range(k):
                    if piece_choice[j] == i[1]:
                        topk[j] += 1
                        break
        for j in range(1,k):
            topk[j] += topk[j-1]
        for j in range(k):
            topk[j] /= len(test)
        return topk

if __name__=="__main__":
    model = CBTemplateCls(USE_LSTM, TEMPLATE_NUM, DEVICE)
    if TRAIN:
        with open("data/TemplateTrain.json") as f:
            tr = json.load(f)
        with open("data/TemplateVal.json") as f:
            va = json.load(f)
        model.fit(tr, va, EPOCH, TOPK)
    else:
        with open("data/TemplateTest.json") as f:
            te = json.load(f)
        topk = model.test(te, TOPK)
        print("topk:", topk)

