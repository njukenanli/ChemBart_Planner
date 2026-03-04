from ChemBart import CB_mul_END
import json
TRAIN = False
c = CB_mul_END("temp_yield_bart","cuda:1")
with open("temp_data.json") as f:
    l = json.load(f)
data = [(i + "<n01><end>", l[i]) for i in l]
length = len(data)
print(length,data[5])
if TRAIN:
    c.single_train(data, 50, int(length*0.8), int(length*0.1), int(length*0.1))
else:
    import pandas as pd
    ans = c.test(data[int(length*0.9):])
    print(ans[0])
    df_temp = pd.DataFrame(ans[1][0])
    df_temp.to_csv("temprature_test.csv")
    df_yiel = pd.DataFrame(ans[1][1])
    df_yiel.to_csv("yield_test.csv")

