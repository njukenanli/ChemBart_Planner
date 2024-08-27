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
    def __init__(self, dev = "cuda:0"):
        self.model = ChemBart(dev)
        self.model.BartNN.eval()

    def share_memory(self):
        self.model.BartNN.share_memory()

    def precursor(self, product, k, lock = None):
        if lock is not None:
            lock.acquire()
        out = self.model.predict("<msk>>>"+product, max_len = 1018, decoder_input="<cls>", top_k = k) 
        if lock is not None:
            lock.release()
        for j in range(len(out)):
            ans = out[j][0]
            if ans[-5:] == "<end>":
                out[j][0] = ans[5:-5]
            elif ans[-1] == ">":
                out[j][0] = ans[5:-1]
            else:
                out[j][0] = ans[5:]
        return out

    def reagent(self, reactant, product, k, lock = None):
        if lock is not None:
            lock.acquire()
        out = self.model.predict(reactant+"><msk>>"+product, max_len = 1018 - len(reactant), decoder_input = "<cls>"+reactant+">", top_k = k)
        if lock is not None:
            lock.release()
        for j in range(len(out)):
            ans = out[j][0]
            if ans[-1] != ">":
                out[j][0] = ans.split(">")[-1]
            else:
                ans = ans.split(">")[-2]
                if ans[-4:] == "<end":
                    ans = ans[:-4]
                out[j][0] = ans
        return out

    def product(self, reactant, reagent, k, lock = None):
        if lock is not None:
            lock.acquire()
        out = self.model.predict(reactant+">"+reagent+"><msk>", max_len = 1018 - len(reactant) - len(reagent), decoder_input = "<cls>" + reactant + ">" + reagent + ">", top_k = k)
        if lock is not None:
            lock.release()
        for j in range(len(out)):
            ans = out[j][0]
            if ans[-1] != ">":
                out[j][0] = ans.split(">")[-1]
            else:
                ans = ans.split(">")[-2]
                if ans[-4:] == "<end":
                    ans = ans[:-4]
                out[j][0] = ans
        return out

class CBRL():
    def __init__(self, dev):
        self.model = CB_MCTS(dev)
        self.tokenizer = self.model.core.tokenizer
        self.model.core.eval()
    
    def share_memory(self):
        self.model.core.share_memory()

    def policy(self, product, precursorlist, lock = None):
        inputlist = [self.tokenizer.encoder(precursor + ">>" + product) for precursor in precursorlist]
        if lock is not None:
            lock.acquire()
        with torch.no_grad():
            ret = (self.model.policy(inputlist)).tolist()
        if lock is not None:
            lock.release()
        return ret

    def value(self, product, lock = None):
        smi = self.tokenizer.encoder(product)
        if lock is not None:
            lock.acquire()
        with torch.no_grad():
            ret = (self.model.value(smi)).item()
        if lock is not None:
            lock.release()
        return ret

