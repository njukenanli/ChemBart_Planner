import os, sys, json
sys.path.append(os.path.dirname(os.path.abspath("__file__"))+"/functions/ChemBart")
from CB_Planner.functions.mcts_base import *
from CB_Planner.functions.nn import pv, ty
from CB_Planner.functions.utils import utils
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force = True)
from torch.multiprocessing import Pool, Manager
import torch

class MY_MCTS(MCTS_BASE):
    end_set = set() # I suggest using dict and set for find operation because in python dict and set are based on Hash table so find operation takes O(1)
    def __init__(self, pv, mcts_times = 200, max_route_len = 16, max_search_depth = 8, debug = False):
        with open("data/basic_mol.json") as f:
            self.end_set = set(json.load(f)) # this is buyable molecule set from Emolecules with chirality
        super().__init__(gen_choice = self.gen_choice, is_end = self.is_end, gen_value = self.gen_value, mcts_times = mcts_times, max_route_len = max_route_len, max_search_depth = max_search_depth, debug = debug)
        self.pv = pv
    def gen_choice(self, status, lock1 = None, lock2 = None, lock3 = None):
        return self.pv.GenChildnodePolicy(status, lock1, lock2, lock3)
    def gen_value(self, status, lock3 = None):
        return self.pv.Value(status, lock3)
    def is_end(self, status):
        return (status in self.end_set)

class CB_Planner():
    def __init__(self, train = False, max_train_data_num = 5000, mcts_times = 200, max_route_len = 16, max_search_depth = 8, debug = False, gen_dev = "cuda:0", val_dev = "cuda:1", rl_dev = "cuda:2", temp_yield_dev = "cuda:0", choiceperstep = 10, process_parallel = True, pool_zise = 4, semaphore_per_model = 2 ):
        self.train = train
        self.mcts_times = mcts_times
        self.max_route_len = max_route_len
        self.max_search_depth = max_search_depth
        self.debug = debug
        self.pool_zise = pool_zise
        self.semaphore_per_model = semaphore_per_model
        self.pv = pv(gen_dev = gen_dev, val_dev = val_dev, rl_dev = rl_dev, choiceperstep = choiceperstep)
        self.temp_yield_dev = temp_yield_dev
        self.process_parallel = process_parallel
        self.max_train_data_num = max_train_data_num
    def paraller_unit(self, target, idx, alternatives, lock1 = None, lock2 = None, lock3 = None):
        target = utils.canonize(target)
        if target is None: return {"answers_"+str(idx)+"_route_0": [1, {target: None}, {target: None}]}
        mcts = MY_MCTS(pv = self.pv, mcts_times = self.mcts_times, max_route_len = self.max_route_len, max_search_depth = self.max_search_depth, debug = self.debug)
        return mcts.play(target, idx, alternatives = alternatives, train = self.train, max_train_data_num = self.max_train_data_num, lock1 = lock1, lock2 = lock2, lock3 = lock3)
    def plan(self, tasklist):
        routes = dict()
        if len(tasklist) == 1 or (not self.process_parallel):
            for i in range(len(tasklist)):
                routes.update(self.paraller_unit(tasklist[i][0], i, tasklist[i][1], None, None, None))
        else:
            self.pv.share_memory()
            mamagerlist = [Manager(), Manager(), Manager()]
            locklist = [i.Semaphore(self.semaphore_per_model) for i in managerlist]
            with Pool(self.pool_size) as pool:
                res = []
                for i in range(len(tasklist)):
                    res.append(pool.apply_async(self.paraller_unit, (tasklist[i][0], i, tasklist[i][1], locklist[0], locklist[1], locklist[2])))
                pool.close()
                pool.join()
                for i in res:
                    routes.update(i.get())
        del self.pv
        # process answer, add other reaction info...
        tymodel = ty(self.temp_yield_dev)
        ty.AddTemperatureYield(routes)
        return routes
    def save_to_file(self, routes):
        for route_name in routes:
            with open("answer/" + route_name, mode = "w+") as f:
                print(route_name, file = f)
                json.dump(routes[route_name][0], f, indent = True)
                print("\nsynthesis route:", file = f)
                json.dump(routes[route_name][1], f, indent = True)
                print("\nreaction information per step:", file = f)
                json.dump(routes[route_name][2], f, indent = True)
        return
