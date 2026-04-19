# procdata.py
import json
import torch
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
class RSZ_processor():
    def __init__(self,data_path):
        self.num = 13312
        with open(data_path) as f:
            self.data = json.load(f)
    def process(self):
        out = []
        for i in self.data:
            ecfp = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(i[0][0][7:]),radius = 2,nBits=2048)).float()
            p = [0.0]*self.num
            for j in range(len(i[1][1])):
                p[i[2][j]] = i[1][1][j]
            p = torch.tensor(p)
            v = torch.tensor([float(i[1][0])])
            out.append([ecfp,(p,v)])
        return out
