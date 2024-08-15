import sys
from ChemBart import ChemBart, CB_mul_END, CB_MCTS
import torch

class CBTempYield():
    def __init__(self, name = "temp_yield_bart", dev = "cuda:0"):
        self.model = CB_mul_END(name, dev)
        self.model.eval()

    def pred(self, reactant, reagant, product):
        smi = reactant + ">" + reagant + ">" + product + "<n01><end>"
        inp = self.model.tokenizer.encoder(smi)
        with torch.no_grad():
            out = self.model(inp)
        return out.tolist()

class CBRetro():
    def __init__(self, dev = "cuda:0", k = 10):
        self.model = ChemBart()
        self.dev = dev
        self.model.BartNN.eval()
        self.k = k

    def share_memory(self):
        self.model.BartNN.share_memory()

    def precursor(self, product, lock = None):
        if lock is not None:
            lock.acquire()
        out = self.model.predict("<msk>>>"+product, max_len=600, decoder_input="<cls>", device=self.dev, top_k = self.k) 
        if lock is not None:
            lock.release()
        ans = []
        for j in out:
            if j[0][-5:] == "<end>":
                ans.append((j[0][5:-5],j[1]))
            elif j[0][-1] == ">":
                ans.append((j[0][5:-1],j[1]))
            else:
                ans.append((j[0][5:],j[1]))
        return ans

    def reagent(self, reactant, product, lock = None):
        if lock is not None:
            lock.acquire()
        o = self.model.predict(reactant+"><msk>>"+product, max_len = 600, decoder_input = "<cls>"+reactant+">", device=self.dev, top_k = self.k)
        if lock is not None:
            lock.release()
        ans = []
        for j in o:
            if j[0][-1] != ">":
                reag = j[0].split(">")[-1]
            else:
                reag = j[0].split(">")[-2]
            if reag[-4:] == "<end":
                reag = reag[:-4]
            ans.append((reag,j[1]))
        return ans

    def product(self, reactant, reagent, lock = None):
        if lock is not None:
            lock.acquire()
        out = self.model.predict(reactant+">"+reagent+"><msk>", max_len = 1018 - len(reactant) - len(reagent), decoder_input = "<cls>" + reactant + ">" + reagent + ">", device=self.dev, top_k = self.k)
        if lock is not None:
            lock.release()
        ans = []
        for j in out:
            if j[0][-1] != ">":
                prod = j[0].split(">")[-1]
            else:
                prod = j[0].split(">")[-2]
            if prod[-4:] == "<end":
                prod = prod[:-4]
            ans.append((prod, j[1]))
        return ans

class RL():
    def __init__(self, dev):
        self.model = CB_MCTS(dev)
        self.tokenizer = self.model.core.tokenizer
        self.dev = torch.device(dev)
        self.model.core.eval()
    
    def share_memory(self):
        self.model.core.share_memory()

    def policy(self, product, precursorlist, lock = None):
        inputlist = [self.tokenizer.encoder(precursor + ">>" + product) for precursor in precursorlist]
        if lock is not None:
            lock.acquire()
        inputlist = [i.to(self.dev) for i in inputlist]
        with torch.no_grad():
            ret = (self.model.policy(inputlist)).tolist()
        if lock is not None:
            lock.release()
        return ret

    def value(self, product, lock = None):
        smi = self.tokenizer.encoder(product)
        if lock is not None:
            lock.acquire()
        smi = smi.to(self.dev)
        with torch.no_grad():
            ret = (self.model.value(smi)).item()
        if lock is not None:
            lock.release()
        return ret

