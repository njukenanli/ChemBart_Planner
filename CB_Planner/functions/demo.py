import torch
from torch import tensor,device
import time
def train(model, rank, lock):
    lock.acquire()
    print("get", flush=True)
    res = model(tensor([[3,2,1]]).to(device("cuda:0"))).item()
    #res = rank
    lock.release()
    print(res, flush = True)
    return res

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath("__file__"))+"/ChemBart")
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force = True)
    from torch.multiprocessing import Pool, Process, Manager
    from ChemBart.ChemBart import CB_END
    dev = device("cuda:0")
    m = Manager()
    lock = m.Semaphore(2)
    model = CB_END(out_type = 1, name = "haha", device = "cuda:0", ran = 0)
    model.to(dev)
    model.eval()
    model.share_memory()
    st = time.time()
    print(model(tensor([[3,2,1]]).to(dev)).item())
    print(time.time() - st, "s")
    with torch.no_grad():
        with Pool(4) as pool:
            res = []
            st = time.time()
            for rank in range(20):
                res.append(pool.apply_async(train, (model, rank, lock)))
            pool.close()
            pool.join()
            print(time.time() - st, "s")
            for i in res:
                print(i.get())

'''
processes = []
for rank in range(2):
    p = Process(target=train, args=(model, lock))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
'''
