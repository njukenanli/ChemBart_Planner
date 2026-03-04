from ChemBart import CB_END
import json
c = CB_END(1,"temperatue","cuda:3",0)
with open("temp_data.json") as f:
    l = json.load(f)
data=[]
for i in l:
    temp = l[i][0]
    if type(temp)==type(1.1):
        data.append((i,temp))
length = len(data)
print(length,data[5])
c.single_train(data, 50, int(length*0.8), int(length*0.1), int(length*0.1))
