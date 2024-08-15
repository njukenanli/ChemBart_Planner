from ChemBart.api import CBTempYield, CBRetro, RL
from mcts_base import NODE
from utils import utils
class pv():
    def __init__(self, gen_dev = "cuda:0", val_dev = "cuda:1", rl_dev = "cuda:2", choiceperstep = 10):
        self.backward = CBRetro(dev = gen_dev, k = choiceperstep)
        self.forward = CBRetro(dev = val_dev, k = choiceperstep)
        self.rl = RL(rl_dev)
        return

    def share_memory(self):
        self.backward.share_memory()
        self.forward.share_memory()
        self.rl.share_memory()
        return

    def GenChildnodePolicy(self, smi, lock1, lock2, lock3):
        '''
        NODE GenChildnodePolicy(string)
        '''
        #SL
        raw_ans = self.backward.precursor(smi, lock1)
        state_dict = dict()
        temp = 0
        for idx in range(len(raw_ans)):
            temp += raw_ans[idx][1]
        for idx in range(len(raw_ans)):
            raw_ans[idx][1] = raw_ans[idx][1]/temp
        for i in raw_ans:
            precursor = utils.canonize(i[0])
            if precursor is not None and precursor != smi:
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
            agt = self.forward.reagent(precursor, smi, lock2)
            agt = utils.canonize(agt, True)
            prod = self.forward.product(precursor, agt, lock2)
            prod = utils.canonize(prod, True)
            if utils.weak_compare(prod, smi):
                childlist.append(precursor)
                reagent.append(agt)
                Plist.append(state_dict[precursor])
            else:
                continue
        Plist = self._normalize(Plist)
        #RL
        policy = self.rl.policy(smi, childlist, lock3)
        for idx in range(len(Plist)):
            Plist[i] = Plist[i]*policy[i]
        Plist = self._normalize(Plist)
        #encapsulate
        node = NODE(childlist = childlist.split("."), reagent = reagent, Plist = Plist, V = None)
        return node
    
    def _normalize(self, l):
        temp = sum(l)
        for idx in range(len(l)):
            l[idx] = l[idx]/temp
        return l

    def Value(self, smi, lock3):
        '''
        float Value(string)
        '''
        return self.rl.value(smi, lock3)

class ty():
    def __init__(self, dev):
        self.model = CBTempYield(name = "temp_yield_bart", dev = dev)
    def AddTemperatureYield(self, routes):
        for route_name in routes:
            for product in routes[route_name][2]:
                routes[route_name][2][product]["temperature"], routes[route_name][2][product]["yield"] = self.model.pred(routes[route_name][2][product]["precursors"], routes[route_name][2][product]["reagents"], product) 
