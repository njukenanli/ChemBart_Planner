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
    def __init__(self, pv, params, basic_mol_path):
        with open(basic_mol_path) as f:
            self.end_set = set(json.load(f)) # this is buyable molecule set from Emolecules with chirality
        self.pv = pv
        super().__init__(gen_choice = self.gen_choice, is_end = self.is_end, gen_value = self.gen_value, **params)
        return
    def gen_choice(self, status):
        return self.pv.GenChildnodePolicy(status)
    def gen_value(self, status):
        return self.pv.Value(status)
    def is_end(self, status):
        res = utils.general_basic_mol(status) or (status in self.end_set)
        return res

class CB_Planner():
    def __init__(self, config):
        self.config = config
    def paraller_unit(self, pv, target, idx, alternatives, train_data_lock = None):
        root = utils.canonize(target)
        if root is None: return {"answers_"+str(idx)+"_"+target+"_route_0": [{"success" : -1 , "probability" : 0}, {target: "wrong smiles!"}, {target: "wrong smiles!"}]}
        mcts = MY_MCTS(pv = pv, params = self.config["mcts"], basic_mol_path = self.config["bascic_mol"])
        return mcts.play(root, idx, file_path = self.config["save_path"], alternatives = alternatives, train = self.config["getdata"]["train"], max_train_data_num = self.config["getdata"]["max_train_data_num"], train_data_lock = train_data_lock)
    def plan(self, tasklist):
        nn_config = self.config["nn"]
        pv = PV(
            chembart_path = self.config["chembart_path"],
            rl_path = self.config["rl_path"],
            gen_dev = nn_config["gen_dev"],
            pro_dev = nn_config["pro_dev"],
            agt_dev = nn_config["agt_dev"],
            rl_dev = nn_config["rl_dev"],
            topk = nn_config["topk"],
            topp = nn_config["topp"],
            sampling_method = nn_config["sampling_method"],
            num_samples = nn_config["num_samples"],
        )
        routes = dict()
        if len(tasklist) == 1 or (not self.config["parallel"]["process_parallel"]):
            for i in range(len(tasklist)):
                routes.update(self.paraller_unit(pv, tasklist[i][0], i, tasklist[i][1]))
        else:
            with Manager() as manager:
                locklist = [manager.Semaphore(self.config["parallel"][key]) for key in (
                    "semaphore_retro_model", "semaphore_reagt_model",
                    "semaphore_prod_model", "semaphore_rl_model",
                )]
                train_data_lock = manager.Lock() if self.config["getdata"]["train"] else None
                pv.share_memory(*locklist)
                with Pool(self.config["parallel"]["pool_size"]) as pool:
                    res = []
                    for i in range(len(tasklist)):
                        res.append(pool.apply_async(self.paraller_unit, (pv, tasklist[i][0], i, tasklist[i][1], train_data_lock)))
                    pool.close()
                    pool.join()
                    for i in res:
                        routes.update(i.get())
        del pv
        # process answer, add other reaction info...
        ty = TY(ty_path = self.config["ty_path"], dev = nn_config["temp_yield_dev"])
        ty.AddTemperatureYield(routes)
        del ty
        return routes
    def save_to_file(self, routes):
        os.makedirs(self.config["save_path"], exist_ok=True)
        for route_name in routes:
            with open(os.path.join(self.config["save_path"], route_name.replace("/", "")), mode = "w+") as f:
                print(route_name, file = f)
                json.dump(routes[route_name][0], f, indent = True)
                print("\nsynthesis route:", file = f)
                json.dump(routes[route_name][1], f, indent = True)
                print("\nreaction information per step:", file = f)
                json.dump(routes[route_name][2], f, indent = True)
        return
