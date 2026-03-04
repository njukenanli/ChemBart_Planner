from ChemBart import CB_Regression
import json
TRAIN = False
LABEL_NUM = 4

task_token = ["<n00>", "<n01>", "<n02>","<n03>", "<n04>"]
ends = "".join(task_token[:LABEL_NUM - 1]) + "<end>"
c = CB_Regression(name = "demo", label_num = LABEL_NUM, device = "cpu") #"cuda:0", "cuda:1" etc.
with open("demo_data.json") as f:
    l = json.load(f)
for i in range(len(l)):
    l[i][0] += ends
print(l[1])
if TRAIN:
    c.fit(data = l, epoch = 1, batch_size = 2, tr=5 , val = 2, te = 2, id_maxlen = 100)
    #data, epoch, batch size, train num, val num, test num, max input length
else:
    out = c.test(l[7:], batch_size = 2, id_maxlen = 100)
    print(out)
