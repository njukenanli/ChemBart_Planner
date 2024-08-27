from ChemBart.api import CBRetro, RL
from mcts_base import NODE
from utils import utils
class pv():
    def __init__(self, gen_dev = "cuda:0", val_dev = "cuda:1", rl_dev = "cuda:2", choiceperstep = 10):
        self.backward = CBRetro(dev = gen_dev, k = choiceperstep)
        self.forward = CBRetro(dev = val_dev, k = choiceperstep)
        self.rl = RL(rl_dev)

    def GenChildnodePolicy(self, smi):
        '''
        NODE GenChildnodePolicy(string)
        '''
        #SL
        raw_ans = self.backward.precursor(smi)
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
            agt = self.forward.reagent(precursor, smi)
            agt = utils.canonize(agt, True)
            prod = self.forward.product(precursor,agt)
            prod = utils.canonize(prod, True)
            if utils.weak_compare(prod, smi):
                childlist.append(precursor)
                reagent.append(agt)
                Plist.append(state_dict[precursor])
            else:
                continue
        Plist = self._normalize(Plist)
        #RL
        policy = self.rl.policy(smi, childlist)
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

    def Value(self, smi):
        '''
        float Value(string)
        '''
        return self.rl.value(smi)
