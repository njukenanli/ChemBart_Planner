from CB_Planner.functions.ChemBart.api import CBTempYield, CBRetro, CBRL
from CB_Planner.functions.mcts_base import NODE
from CB_Planner.functions.utils import utils
import torch
class PV():
    def __init__(self,chembart_path, rl_path, gen_dev = "cuda:0", pro_dev = "cuda:1", agt_dev = "cuda:2", rl_dev = "cuda:3", topk = 10, topp = 0.9, sampling_method='beam', num_samples=50):
        self.backward = CBRetro(path = chembart_path, dev = gen_dev)
        self.agent = CBRetro(path = chembart_path, dev = agt_dev) # for validation
        self.forward = CBRetro(path = chembart_path, dev = pro_dev)
        self.rl = CBRL(path = rl_path, dev = rl_dev)
        self.sampling_method = sampling_method
        self.k = topk
        self.p = topp
        self.num_samples = num_samples
        self.lock1 = None
        self.lock2 = None
        self.lock3 = None
        self.lock4 = None
        return

    def share_memory(self, lock1, lock2, lock3, lock4, strategy = "file_system"):
        torch.multiprocessing.set_sharing_strategy(strategy)
        self.backward.share_memory()
        self.agent.share_memory()
        self.forward.share_memory()
        self.rl.share_memory()
        self.lock1 = lock1
        self.lock2 = lock2
        self.lock3 = lock3
        self.lock4 = lock4
        return

    
    def GenChildnodePolicy(self, smi):
        '''
        NODE GenChildnodePolicy(string)
        '''
        #SL
        raw_ans = self.backward.precursor(smi, self.lock1, sampling_method=self.sampling_method, num_samples=self.num_samples, top_k=self.k, top_p=self.p)
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
            agt = self.agent.reagent(precursor, smi, 1, self.lock2)[0][0]
            agt = utils.canonize(agt, True)
            prodlist = self.forward.product(precursor, agt, 3, self.lock3)
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
            policy = self.rl.policy(smi, childlist, self.lock4)
            for idx in range(len(Plist)):
                Plist[idx] = Plist[idx]*policy[idx]
            self._normalize(Plist)
        #encapsulate
        for idx in range(len(childlist)):
            childlist[idx] = childlist[idx].split(".")
        node = NODE(childlist = childlist, reagent = reagent, Plist = Plist, V = None)
        return node
    
    def GenChildnodePolicy_more_mol(self, smi):
        '''
        NODE GenChildnodePolicy(string)
        Simplified version: skip forward validation step
        '''
        # Step 1: 使用语言模型生成初始前体候选
        raw_ans = self.backward.precursor(smi, self.lock1, sampling_method=self.sampling_method, num_samples=self.num_samples, top_k=self.k, top_p=self.p)

        if not raw_ans:
            return NODE(childlist=[], reagent=[], Plist=[], V=None)

        state_dict = {}

        # Step 2: 归一化并去重合并相同结构
        temp = sum(raw_ans[i][1] for i in range(len(raw_ans)))
        for idx in range(len(raw_ans)):
            raw_ans[idx][1] = raw_ans[idx][1] / temp

        for i in raw_ans:
            precursor = utils.canonize(i[0])
            if (precursor is not None) and (not utils.weak_compare(precursor, smi)):
                if precursor in state_dict:
                    state_dict[precursor] += i[1]
                else:
                    state_dict[precursor] = i[1]

        childlist = []
        reagent = []
        Plist = []

        # Step 3: 使用 agent.reagent() 预测试剂
        for precursor in state_dict:
            agt = self.agent.reagent(precursor, smi, 1, self.lock2)[0][0]
            agt = utils.canonize(agt, True)
            childlist.append(precursor)
            reagent.append(agt)
            Plist.append(state_dict[precursor])

        # Step 4: 再次归一化
        self._normalize(Plist)

        if childlist:
            policy = self.rl.policy(smi, childlist, self.lock4)
            for idx in range(len(Plist)):
                Plist[idx] *= policy[idx]
            self._normalize(Plist)

        # Step 5: 构造并返回 NODE 对象
        for idx in range(len(childlist)):
            childlist[idx] = childlist[idx].split(".")  # 分割多个前体

        node = NODE(childlist=childlist, reagent=reagent, Plist=Plist, V=None)
        return node

    def _normalize(self, l):
        temp = sum(l)
        for idx in range(len(l)):
            l[idx] = l[idx]/temp

    def Value(self, smi):
        '''
        float Value(string)
        '''
        return self.rl.value(smi, self.lock4)

class TY():
    def __init__(self, ty_path, dev):
        self.model = CBTempYield(ty_path, name = "temp_yield_bart", dev = dev)
    def AddTemperatureYield(self, routes):
        for route_name in routes:
            for product in routes[route_name][2]:
                try:
                    routes[route_name][2][product]["temperature"], routes[route_name][2][product]["yield"] = self.model.pred(routes[route_name][2][product]["precursors"], routes[route_name][2][product]["reagents"], product) 
                except Exception as e:
                    # 打印错误信息及出错的数据
                    precursors = routes[route_name][2][product]["precursors"]
                    reagents = routes[route_name][2][product]["reagents"]
                    print(f"Error processing {route_name}, {product} with precursors: {precursors}, reagents: {reagents}: {e}")
                    print(f"Failed routes data: {routes}")


class Precursor():
    def __init__(self,chembart_path, rl_path, gen_dev = "cuda:0", pro_dev = "cuda:1", agt_dev = "cuda:2", rl_dev = "cuda:3", topk = 10, topp = 0.9, sampling_method='beam', num_samples=50):
        self.backward = CBRetro(path = chembart_path, dev = gen_dev)
        self.sampling_method = sampling_method
        self.k = topk
        self.p = topp
        self.num_samples = num_samples
        self.lock1 = None
        self.lock2 = None
        self.lock3 = None
        self.lock4 = None
        return

    def share_memory(self, lock1, lock2, lock3, lock4, strategy = "file_system"):
        torch.multiprocessing.set_sharing_strategy(strategy)
        self.backward.share_memory()
        self.lock1 = lock1
        self.lock2 = lock2
        self.lock3 = lock3
        self.lock4 = lock4
        return

    def GenChildnodePolicy_more_mol(self, smi):
        '''
        NODE GenChildnodePolicy(string)
        Simplified version: skip forward validation step
        '''
        # Step 1: 使用语言模型生成初始前体候选
        raw_ans = self.backward.precursor(smi, self.lock1, sampling_method=self.sampling_method, num_samples=self.num_samples, top_k=self.k, top_p=self.p)

        if not raw_ans:
            return NODE(childlist=[], reagent=[], Plist=[], V=None)

        state_dict = {}

        # Step 2: 归一化并去重合并相同结构
        temp = sum(raw_ans[i][1] for i in range(len(raw_ans)))
        for idx in range(len(raw_ans)):
            raw_ans[idx][1] = raw_ans[idx][1] / temp

        for i in raw_ans:
            precursor = utils.canonize(i[0])
            if (precursor is not None) and (not utils.weak_compare(precursor, smi)):
                if precursor in state_dict:
                    state_dict[precursor] += i[1]
                else:
                    state_dict[precursor] = i[1]

        childlist = []
        Plist = []

        # Step 3: 使用 agent.reagent() 预测试剂
        for precursor in state_dict:
            childlist.append(precursor)
            Plist.append(state_dict[precursor])

        # Step 5: 构造并返回 NODE 对象
        for idx in range(len(childlist)):
            childlist[idx] = childlist[idx].split(".")  # 分割多个前体

        node = NODE(childlist=childlist, reagent=None, Plist=Plist, V=None)
        return node

    def _normalize(self, l):
        temp = sum(l)
        for idx in range(len(l)):
            l[idx] = l[idx]/temp
