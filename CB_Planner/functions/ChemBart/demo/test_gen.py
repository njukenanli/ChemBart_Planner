from rdkit import Chem
name = "precursor"
def proc_smi(s):
    l = s.split(".")
    for i in range(len(l)):
        if l[i] == "":
            continue
        l[i] = Chem.MolFromSmiles(l[i])
        if l[i] is not None:
            l[i] = Chem.MolToSmiles(l[i], canonical=True)
    return l
def check_acc(l):
    all_ = len(l)
    err = 0
    for i in l:
        if i is None:
            err += 1
    return (all_, err)
def compare(l1,l2):
    for i in l1:
        for j in l2:
            if (i is not None)\
                    and (j is not None)\
                    and (i == j):
                return True
    return False
with open(name + ".txt") as f:
    l = f.read().splitlines()
for i in range(len(l)):
    l[i] = eval(l[i])
acc = [0]*10
grammar_err = [0]*10
grammar_all = [0]*10
for i in l:
    tru = proc_smi(i[1])
    for j in range(len(i[0])):
        pre = proc_smi(i[0][j][0])
        all_,err = check_acc(pre)
        grammar_all[j] += all_
        grammar_err[j] += err
        if compare(pre,tru):
            acc[j]+=1
            break
with open(name + "_result.txt", mode="a+") as f:
    print(len(l),acc,grammar_err,grammar_all, file = f, sep = "\n", flush=True)
