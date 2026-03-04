from ChemBert import CB_END
import json
c = CB_END(1,"ts_ene_bert","cuda:2",10)
with open("tsdata.json") as f:
    l = json.load(f)
length = len(l)
c.single_train(l, 100, int(length*0.8), int(length*0.1), int(length*0.1))
