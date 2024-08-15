from ChemBert import CB_mul_END
import json
c = CB_mul_END("temp_yield_bert","cuda:1")
with open("temp_data.json") as f:
    l = json.load(f)
data = [("<cls><n01>" + i, l[i]) for i in l]
length = len(data)
print(length,data[5])
c.single_train(data, 50, int(length*0.8), int(length*0.1), int(length*0.1))
