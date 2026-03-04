import ChemBart
import os
import json
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
c=ChemBart.ChemBart()
if True:
    with open("complete_train_string.json") as f:
        stringlist=json.load(f)
else:
    stringlist=["CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)"]
c.paral_train(stringlist)
