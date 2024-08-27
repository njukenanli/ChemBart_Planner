import os, sys, json
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/functions/ChemBart")
from CB_Planner.functions.mcts_base import MCTS_BASE
from CB_Planner.functions.nn import PV, TY
from CB_Planner.functions.utils import utils
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force = True)
from torch.multiprocessing import Pool, Manager
import torch
from copy import deepcopy

class MY_MCTS(MCTS_BASE):
    # I suggest using dict and set for find operation because in python dict and set are based on Hash table so find operation takes O(1)
    def __init__(self, pv, params):
        with open("CB_Planner/data/basic_mol.json") as f:
            self.end_set = set(json.load(f)) # this is buyable molecule set from Emolecules with chirality
        self.pv = pv
        super().__init__(gen_choice = self.gen_choice, is_end = self.is_end, gen_value = self.gen_value, **params)
        return
    def gen_choice(self, status, lock1 = None, lock2 = None, lock3 = None):
        return self.pv.GenChildnodePolicy(status, lock1, lock2, lock3)
    def gen_value(self, status, lock3 = None):
        return self.pv.Value(status, lock3)
    def is_end(self, status):
        res = utils.general_basic_mol(status) or (status in self.end_set)
        return res

class CB_Planner():
    def __init__(self, config):
        self.config = config
    def paraller_unit(self, pv, target, idx, alternatives, lock1 = None, lock2 = None, lock3 = None):
        root = utils.canonize(target)
        if root is None: return {"answers_"+str(idx)+"_"+target+"_route_0": [{"success" : -1 , "probability" : 0}, {target: "wrong smiles!"}, {target: "wrong smiles!"}]}
        mcts = MY_MCTS(pv = pv, params = self.config["mcts"])
        return mcts.play(root, idx, alternatives = alternatives, train = self.config["getdata"]["train"], max_train_data_num = self.config["getdata"]["max_train_data_num"], lock1 = lock1, lock2 = lock2, lock3 = lock3)
    def plan(self, tasklist):
        pv = PV(gen_dev = self.config["nn"]["gen_dev"], val_dev = self.config["nn"]["val_dev"], rl_dev = self.config["nn"]["rl_dev"], choiceperstep = self.config["nn"]["choiceperstep"])
        routes = dict()
        if len(tasklist) == 1 or (not self.config["parallel"]["process_parallel"]):
            for i in range(len(tasklist)):
                routes.update(self.paraller_unit(pv, tasklist[i][0], i, tasklist[i][1], None, None, None))
        else:
            pv.share_memory()
            managerlist = [Manager(), Manager(), Manager()]
            locklist = [i.Semaphore(self.config["parallel"]["semaphore_per_model"]) for i in managerlist]
            with Pool(self.config["parallel"]["pool_size"]) as pool:
                res = []
                for i in range(len(tasklist)):
                    res.append(pool.apply_async(self.paraller_unit, (pv, tasklist[i][0], i, tasklist[i][1], locklist[0], locklist[1], locklist[2])))
                pool.close()
                pool.join()
                for i in res:
                    routes.update(i.get())
        del pv
        # process answer, add other reaction info...
        ty = TY(self.config["nn"]["temp_yield_dev"])
        ty.AddTemperatureYield(routes)
        del ty
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
