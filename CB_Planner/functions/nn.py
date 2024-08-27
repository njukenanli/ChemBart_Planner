from CB_Planner.functions.ChemBart.api import CBTempYield, CBRetro, CBRL
from CB_Planner.functions.mcts_base import NODE
from CB_Planner.functions.utils import utils
import torch
class PV():
    def __init__(self, gen_dev = "cuda:0", val_dev = "cuda:1", rl_dev = "cuda:2", choiceperstep = 10):
        self.backward = CBRetro(dev = gen_dev)
        self.forward = CBRetro(dev = val_dev) # for validation
        self.rl = CBRL(rl_dev)
        self.k = choiceperstep
        return

    def share_memory(self, strategy = "file_system"):
        torch.multiprocessing.set_sharing_strategy(strategy)
        self.backward.share_memory()
        self.forward.share_memory()
        self.rl.share_memory()
        return

    def GenChildnodePolicy(self, smi, lock1, lock2, lock3):
        '''
        NODE GenChildnodePolicy(string)
        '''
        #SL
        raw_ans = self.backward.precursor(smi, self.k, lock1)
        state_dict = dict()
        temp = 0
        for idx in range(len(raw_ans)):
            temp += raw_ans[idx][1]
        for idx in range(len(raw_ans)):
            raw_ans[idx][1] = raw_ans[idx][1]/temp
        for i in raw_ans:
            precursor = utils.canonize(i[0])
            if (precursor is not None) and (not utils.weak_compare(precursor, smi)):
                if precursor in state_dict:
                    state_dict[precursor] += i[1]
                else:
                    state_dict[precursor] = i[1]
            else:
                continue
        childlist = []
        reagent = []
        Plist = []
        for precursor in state_dict:
            agt = self.forward.reagent(precursor, smi, 1, lock2)[0][0]
            agt = utils.canonize(agt, True)
            prodlist = self.forward.product(precursor, agt, 5, lock2)
            for prod, prob in prodlist:
                prod = utils.canonize(prod, True)
                if (prod is not None) and utils.weak_compare(prod, smi):
                    childlist.append(precursor)
                    reagent.append(agt)
                    Plist.append(state_dict[precursor])
                    break
        self._normalize(Plist)
        #RL
        if childlist:
            policy = self.rl.policy(smi, childlist, lock3)
            for idx in range(len(Plist)):
                Plist[idx] = Plist[idx]*policy[idx]
            self._normalize(Plist)
        #encapsulate
        for idx in range(len(childlist)):
            childlist[idx] = childlist[idx].split(".")
        node = NODE(childlist = childlist, reagent = reagent, Plist = Plist, V = None)
        return node
    
    def _normalize(self, l):
        temp = sum(l)
        for idx in range(len(l)):
            l[idx] = l[idx]/temp

    def Value(self, smi, lock3):
        '''
        float Value(string)
        '''
        return self.rl.value(smi, lock3)

class TY():
    def __init__(self, dev):
        self.model = CBTempYield(name = "temp_yield_bart", dev = dev)
    def AddTemperatureYield(self, routes):
        for route_name in routes:
            for product in routes[route_name][2]:
                routes[route_name][2][product]["temperature"], routes[route_name][2][product]["yield"] = self.model.pred(routes[route_name][2][product]["precursors"], routes[route_name][2][product]["reagents"], product) 
