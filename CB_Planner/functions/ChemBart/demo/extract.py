import json
import torch
from CBTokenizer import CBTokenizer
from joblib import Parallel, delayed
with open('outnomap.json') as f:
    l=json.load(f)
def proc(l):
    outs=[]
    outl=[]
    t=CBTokenizer()
    for i in l:
        try:
            templ=[]
            templ.extend(i['conditions'][0])
            templ.extend(i['conditions'][1])
            s=i['reactants']+'>'+'.'.join(templ)+'>'+i['products']
            outs.append(s)
            #outl.extend(t.gen_train_data(s))
        except Exception as err:
            print(s,err)
    return (outs,outl)
lenl=len(l)
lenper = int(lenl/20)
ret = Parallel(n_jobs=20)(delayed(proc)(l[lenper*i:lenper*(i+1)]) for i in range(20))
#outl=[]
outs=[]
for i in ret:
    outs.extend(i[0])
    #outl.extend(i[1])
#for i in l[0:60]:
#    q=proc(i,t)
with open('reaction_string',mode="w+") as f:
    json.dump(outs,f)
#torch.save(outl,"training_data")


