from typing import *
import gc
from queue import Queue
import heapq
import json
import os
from copy import deepcopy

class NODE():
    childlist = []  
    reagent = []
    N: int = 0  # visit times
    V: Optional[int] # Value
    Nlist: List[int]
    qlist: List[float]
    Plist: List[float]
    #if is leaf, V = v from gen_value method, else V = G = mean(Vi for i in children_chosen)
    choice_len: int = 0
    def __init__(self, childlist, reagent, Plist: List[float], V: Optional[int]):
        self.childlist = childlist
        self.reagent = reagent
        self.Plist = Plist
        self.V = V
        self.N = 0
        self.choice_len = len(childlist)
        self.Nlist = [0] * self.choice_len
        self.qlist = [0.0] * self.choice_len
        return
    def __str__(self):
        return "childlist: {}, Nlist: {}\n".format(self.childlist, self.Nlist)

class ROUTE():
    success: int # in {-1, 1}
    states_to_solve : set # for next iteration, use set to avoid duplication
    route : dict
    info : dict
    mean_prob :float # in [0, 1]
    route_len : int 
    train_data: list # in train mode, intermediate policy and value would be retained for RL use
    root: str
    maxlen: int
    is_end: Callable
    def __init__(self, root, is_end, maxlen: int = 20):
        self.states_to_solve = set([root])
        self.root = root
        self.success = -1
        self.route = dict()
        self.info = dict()
        self.mean_prob = 1.0
        self.route_len = 0
        self.train_data = []
        self.maxlen = maxlen
        self.is_end = is_end
    def deepcopy(self):
        newroute = ROUTE(self.root, self.is_end, self.maxlen)
        newroute.states_to_solve = deepcopy(self.states_to_solve)
        newroute.success = self.success
        newroute.route = deepcopy(self.route)
        newroute.info = deepcopy(self.info)
        newroute.mean_prob = self.mean_prob
        newroute.route_len = self.route_len
        newroute.train_data = deepcopy(self.train_data)
        return newroute
    def __lt__(self, other):
        return self.mean_prob >= other.mean_prob
    def __str__(self):
        return "success state: {}, probability: {}, states_to_solve: {}, route_dict: {}\n"\
                .format(self.success, self.mean_prob, self.states_to_solve, self.route)
    def gen_tree(self):
        return {self.root: self._trace(self.root, 0)}
    def _trace(self, root, depth):
        if root not in self.route:
            if self.is_end(root):
                return "basic_molecule"
            else:
                return "failed"
        elif depth > self.maxlen:
            return "failed"
        else:
            return dict([[i, self._trace(i, depth+1)] for i in self.route[root]])

class MCTS_BASE():
    '''
    Users can define their own child class inherited from MCTS_BASE
    '''
    temp_coef: float = 1.0 # this is tau
    Cpuct: float = 1.0
    max_route_len: int = 15
    max_search_depth: int = 8
    mcts_times: int = 400
    update_method: str = 'avg'
    search_space = dict() # map<smiles : node>
    NN_value = dict() # map<smiles : value by NN> 
    debug: bool = False # print process
    gen_choice: Callable # function defined by user in the child class
    #input current stautus
    # return; (value function for current status, 
    #([child_node1[branch1,2,3...],child_node2,3...],[child node probability distribution p1,p2,p3...]))
    gen_value: Optional[Callable] = None # function defined by user in the child class
    #if your method would yield v and p together, then let gen_value = None
    #else set value function for current status in the return of gen_choice to be None and 
    is_end: Callable #bool is_end(current_status)
    class RouteRepresenter():
        success: int 
        prob : float
        route_len : int
        route_idx : int
        childnode_idx : list
        states_to_solve : set
        def __init__(self, prob, route_idx, route_len = -1):
            self.success = -1
            self.prob = prob
            self.route_idx = route_idx
            self.route_len = route_len
            self.childnode_idx = []
            self.states_to_solve = set()
        def __lt__(self, other):
            return self.prob >= other.prob
        def gen_route(self, oldroute, returninfo, search_space):
            newroute = oldroute.deepcopy()
            newroute.success = self.success
            newroute.mean_prob = self.prob
            if self.route_len > 0:
                newroute.route_len = self.route_len
            newroute.states_to_solve = self.states_to_solve
            for i in range(len(self.childnode_idx)):
                father = returninfo[i][0]
                newroute.route[father] = search_space[father].childlist[self.childnode_idx[i]]
                status = (sum([int(oldroute.is_end(mol)) for mol in newroute.route[father]]) == len(newroute.route[father]))
                newroute.info[father] = {"reagents": search_space[father].reagent[self.childnode_idx[i]], "precursors": ".".join(newroute.route[father]), "probability": 1.0 if status else returninfo[i][1][self.childnode_idx[i]]}
            return newroute
    class PrioritizedItem():
        parent_idx: int # position in return_list[route_idx]
        idx: int # position in Plist/childlist
        prob: float # probability, key
        def __init__(self, parent_idx, idx, prob):
            self.parent_idx = parent_idx
            self.idx = idx
            self.prob = prob
        def __lt__(self, other):
            return self.prob >= other.prob
    def __init__ (self, gen_choice: Callable, is_end: Callable, gen_value: Optional[Callable] = None,
            temp_coef: float = 1.0, max_route_len: int = 15, max_search_depth: int = 8, mcts_times: int = 400,
            Cpuct: float = 1.0, update_method: str = 'avg', debug: bool = False):
        self.temp_coef = temp_coef
        self.Cpuct = Cpuct
        self.max_route_len = max_route_len
        self.max_search_depth = max_search_depth
        self.mcts_times = mcts_times
        self.gen_choice = gen_choice
        self.gen_value = gen_value
        self.is_end = is_end
        self.update_method = update_method
        self.debug = debug
        self.search_space = dict()
        self.NN_value = dict()
        return
    def _gen_U(self, p: float, sumn: int, nsa: int) -> float: 
        '''
        protected: this function cannot be accessed by user, but can be inherited by child class
        Users can use their own evaluation method to rewrite these in the child class.
        '''
        return self.Cpuct * p * (sumn**0.5) / (1+nsa)
    def _gen_Q(self, oldQ: float, nsa: int, Gs) -> float: 
        '''
        protected
        Users can use their own evaluation method to rewrite these in the child class.
        '''
        return oldQ + (Gs-oldQ)/nsa
    def _gen_V(self, vlist):
        '''
        protected
        This function counts the V(some paper call it G) of a parent node 
        from all the chilren of one choice from the parent node.
        You can use 'avg' method or 'min' method, or rewrite the function in the child class.
        Other factors may also be considered in the evaluation of V in the child class:
        if so, _dfs method may also need to be rewritten...
        '''
        for i in vlist:
            if i == -1:
                return -1
        if (self.update_method == 'avg'):
            return sum(vlist)/len(vlist)
        elif (self.update_method == 'min'):
            return min(vlist)
        else:
            raise("unknoen update_method type: in the MCTS base class, only avg and min method can be chosen")
    def _argmax(self, l):
        '''
        protected
        '''
        maxvalue = l[0]
        maxindex = 0
        for i in range(1,len(l)):
            if l[i] > maxvalue:
                maxvalue = l[i]
                maxindex = i
        return maxindex

    def _discard(self, routelist):
            q = Queue(maxsize = -1) #BFS
            retained = set()
            for route in routelist:
                for s in route.states_to_solve:
                    q.put(s)
            while (not q.empty()):
                nodev = q.get()
                if nodev not in retained:
                    retained.add(nodev)
                    if nodev in self.search_space:
                        for branches in self.search_space[nodev].childlist:
                            for next_state in branches:
                                q.put(next_state)
            oldkeys = list(self.search_space.keys())
            for key in oldkeys:
                if key not in retained:
                    self.search_space.pop(key)
            del oldkeys, retained
            gc.collect()
            return

    def play(self, root, idx :int, file_path :str, alternatives: int = 1, train: bool = True, 
            max_train_data_num: int = 5000):
        '''
        public: This is the MCTS entrance for the user.
        return;
        shape: number of alternative routes * each route/tree expressed as generalized table
        [answer1{root: {node1@layer1: {children of node1...}, node2@layer1: {} }} , answer2 ...]
        If alternatives == k>1 the program would generate k routes with the beam search.
        I suggest letting alternatives = 1 while training for RL.
        '''
        ret = dict()
        #Step1 varify input need to be solved
        if self.is_end(root):
            return {"answer_"+str(idx)+"_"+root+"_route_0": [{"success": 1, "probablity": 1.0}, {root: "basic mol"}, {root: "basic mol"}]}
        
        #Step2 prepare to solve
        routelist = [ROUTE(root, self.is_end, self.max_route_len)] # for multi_route design it would be expanded
        finished_route = []
        
        #Step3 solve for each iteration
        for iteration in range(self.max_route_len):
            return_list = []
            for a in range(len(routelist)):
                return_list.append([self._dfs_main(status) for status in routelist[a].states_to_solve])
            
            #Step4 analyse these multiple routes computed and adopt beam search
            # return_list dim : 
            #   [dim1: answer of each route;
            #   dim2: answer of each branch in this route;
            #   dim3: (root, policy) for this branch]
            new_route_list = [] # for next iteration, item: RouteRepresenter
            for a in range(len(routelist)):
                #for each route 
                if routelist[a].mean_prob == 0.0:
                    new_route_list.append(self.RouteRepresenter(0.0, a))
                    continue 
                failed = False
                for last_state in return_list[a]:
                    if not self.search_space[last_state[0]].childlist:
                        failed = True
                        break
                if failed:
                    new_route_list.append(self.RouteRepresenter(0.0, a))
                    continue  
                if train:
                    routelist[a].train_data.extend([[[last_state[0], self.search_space[last_state[0]].childlist],[0, last_state[1]]] for last_state in return_list[a]])

                #go to beam search
                beam_count = alternatives
                heaplist = [] 
                # adopt a priority queue (heap) to find top-kth for each branch
                picked_policy_list = [] # used to pick top-kth route for the next round
                for last_state_id in range(len(return_list[a])):
                    heaplist.append([self.PrioritizedItem(last_state_id, idx, return_list[a][last_state_id][1][idx])  \
                            for idx in range(self.search_space[return_list[a][last_state_id][0]].choice_len)])
                    heapq.heapify(heaplist[-1])
                newroute = self.RouteRepresenter(routelist[a].mean_prob, a, routelist[a].route_len)
                prob_prod = newroute.prob ** newroute.route_len
                for last_state_id in range(len(return_list[a])):
                    father = return_list[a][last_state_id][0]
                    item = heapq.heappop(heaplist[last_state_id])
                    newroute.childnode_idx.append(item.idx)
                    branch_list = [] #save states to solve
                    for next_state in self.search_space[father].childlist[item.idx]:
                        if (not self.is_end(next_state)):
                            newroute.states_to_solve.add(next_state)
                            branch_list.append(next_state)
                    newroute.route_len += 1
                    if branch_list:
                        prob_prod *= item.prob
                        picked_policy_list.append((item.prob, branch_list))
                    else:
                        picked_policy_list.append((1.0, branch_list))
                if not newroute.states_to_solve:
                    newroute.success = 1
                newroute.prob = prob_prod**(1/newroute.route_len)
                beam_count -= 1
                new_route_list.append(newroute)
                # if it is for single path, then this while iteration will be skipped
                top_nth_heap = []
                while (beam_count > 0):
                    if not top_nth_heap:
                        for last_state_id in range(len(return_list[a])):
                            if not heaplist[last_state_id]:
                                continue
                            top_nth_heap.append(heapq.heappop(heaplist[last_state_id]))
                        if not top_nth_heap:
                            break #if there's noting to add, we cannot collect enough routes, just break
                        heapq.heapify(top_nth_heap)
                    item = heapq.heappop(top_nth_heap) 
                    # for the 2nd best choice, we substitute only 1 branch with the max 2nd largest value
                    newroute = deepcopy(new_route_list[-1])
                    newroute.success = -1
                    newroute.childnode_idx[item.parent_idx] = item.idx
                    branch_list = []
                    for next_state in self.search_space[return_list[a][item.parent_idx][0]].childlist[item.idx]:
                        if not self.is_end(next_state):
                            branch_list.append(next_state)
                    #print(picked_policy_list[item.parent_idx])
                    #prob_prod = prob_prod / picked_policy_list[item.parent_idx][0]
                    prob_prod = prob_prod / (picked_policy_list[item.parent_idx][0] + 1e-8)
                    if branch_list:
                        prob_prod *= return_list[a][item.parent_idx][1][item.idx]
                        picked_policy_list[item.parent_idx] = (return_list[a][item.parent_idx][1][item.idx], branch_list)
                    else:
                        picked_policy_list[item.parent_idx] = (1.0, branch_list)
                    newroute.states_to_solve = set()
                    for next_state in picked_policy_list:
                        newroute.states_to_solve.update(next_state[1])
                    if not newroute.states_to_solve:
                        newroute.success = 1
                    newroute.prob = prob_prod**(1/newroute.route_len)
                    beam_count -= 1
                    new_route_list.append(newroute)
            beam_count = alternatives
            oldroutelist = routelist
            routelist = []
            heapq.heapify(new_route_list)
            while (beam_count and new_route_list):
                item = heapq.heappop(new_route_list)
                if item.success == 1:
                    finished_route.append(item.gen_route(oldroutelist[item.route_idx], return_list[item.route_idx], self.search_space))
                    alternatives -= 1
                else:
                    routelist.append(item.gen_route(oldroutelist[item.route_idx], return_list[item.route_idx], self.search_space))
                beam_count -= 1
            del oldroutelist

            
            #(Optional) print intermediate process
            if (self.debug):
                print("iteration", iteration)
                print("search_space:")
                for item in self.search_space:
                    print(item,":",self.search_space[item])
                print("routelist:")
                for item in routelist:
                    print(item)
                print("finished routelist:")
                for item in finished_route:
                    print(item)
            if (not alternatives):
                break
            if len(routelist) == 0 or routelist[0].mean_prob == 0.0:
                break

            #Step5 garbage collection: discard branches not selected
            self._discard(routelist)
        
        #Step6 save output route and (Optional) RL train data
        if alternatives > 0:
            finished_route.extend(routelist[0:alternatives])
        for route_idx in range(len(finished_route)):
            ans = finished_route[route_idx].gen_tree()
            ret["answer_"+str(idx)+"_"+root+"_route_"+str(route_idx)] = [{"success": finished_route[route_idx].success, "probability": finished_route[route_idx].mean_prob}, ans, finished_route[route_idx].info]
        train_data = []
        for route in finished_route:
            for train_data_idx in range(len(route.train_data)):
                route.train_data[train_data_idx][1][0] = route.success
            train_data.extend(route.train_data)

        # 储存mcts过程路径
        save_path = os.path.join(file_path, "train_data.json")  # 使用标准路径拼接


        # 创建目标目录（自动处理多级目录）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保父目录存在
        
        # 读取已有数据（如果存在）
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                existing_data = json.load(f)
                train_data.extend(existing_data)
        
        # 写入新数据（限制最大数量）
        with open(save_path, 'w+') as f:
            json.dump(
                #train_data[:self.config["getdata"]["max_train_data_num"]],  # 应用数据量限制
                train_data,
                f,
                indent=2  # 添加缩进提高可读性
            )
            print(f"成功保存训练数据到：{os.path.abspath(save_path)}")  # 打印绝对路径确认
        return ret

    def _check_cycle(self, childlist, ancestors): 
        for i in childlist:
            if i in ancestors: 
                return True
        return False
 
    def _dfs_main(self, root):  
        ancestor = set()
        
        # === 打印根节点初始状态 ===
        if root not in self.search_space:
            node = self.gen_choice(root)
            print(f"\n[DFS_MAIN] Starting MCTS for root: {root}")
            print(f"         Total branches: {len(node.childlist)}")
            for i, (childs, p) in enumerate(zip(node.childlist, node.Plist)):
                # 获取初始价值（用于调试）
                v_init = self.gen_value(childs[0]) if childs else -1  # 取第一个前体估计
                print(f"  Branch {i}: {childs} | P={p:.3f} | V_init≈{v_init:.3f}")
        # =========================

        # 执行 mcts_times 次模拟
        for count in range(self.mcts_times):
            self._dfs(root, 1, ancestor)
        
        # 生成最终策略
        policy = [0.0] * self.search_space[root].choice_len
        for i in range(self.search_space[root].choice_len):
            policy[i] = self.search_space[root].Nlist[i] ** (1/self.temp_coef)
        norm = sum(policy)
        policy = [p / norm for p in policy]

        # === 打印最终统计 ===
        print(f"\n[DFS_MAIN] MCTS completed for {root} (mcts_times={self.mcts_times})")
        for i in range(self.search_space[root].choice_len):
            n = self.search_space[root].Nlist[i]
            q = self.search_space[root].qlist[i]
            p = self.search_space[root].Plist[i]
            print(f"  Branch {i}: N={n:3d} | Q={q:.3f} | P={p:.3f} | Policy={policy[i]:.3f}")
        # =====================

        return (root, policy)
    
    def _dfs(self, root: str, depth: int, ancestors): 
        if root not in self.search_space:
            self.search_space[root] = self.gen_choice(root)
        self.search_space[root].N += 1

        if not self.search_space[root].choice_len:
            self.search_space[root].V = -1
            return

        # === Select: 打印 PUCT 计算过程 ===
        maxPUCT = -1e9
        maxindex = -1
        puct_scores = []
        
        print(f"\n[DFS] Depth={depth} | Node: {root} | Total visits: {self.search_space[root].N}")
        print(f"      {'Idx':<3} {'Precursors':<30} {'P':<6} {'N':<4} {'Q':<6} {'U':<6} {'PUCT':<6}")
        print(f"      {'-'*60}")

        sum_n = self.search_space[root].N
        for i in range(self.search_space[root].choice_len):
            childs = self.search_space[root].childlist[i]
            p = self.search_space[root].Plist[i]
            n_sa = self.search_space[root].Nlist[i]
            q = self.search_space[root].qlist[i]

            # 检查循环
            if self._check_cycle(childs, ancestors): 
                print(f"      {i:<3} {'(cycle) ' + '.'.join(childs):<30} {p:<6.3f} {n_sa:<4} {q:<6.3f} {'-':<6} {'-':<6}")
                continue

            # 计算 U 和 PUCT
            U = self._gen_U(p, sum_n, n_sa)
            PUCT = q + U
            puct_scores.append((i, PUCT, p, n_sa, q, U))

            if PUCT > maxPUCT:
                maxPUCT = PUCT
                maxindex = i

            # 打印每一行
            precursors = '.'.join(childs) if childs else '[]'
            print(f"      {i:<3} {precursors:<30} {p:<6.3f} {n_sa:<4} {q:<6.3f} {U:<6.3f} {PUCT:<6.3f}")

        if maxindex == -1: 
            self.search_space[root].V = -0.99
            print(f"      [!] All branches skipped (cycle or invalid)")
            return 
        # ===================================

        next_state_list = self.search_space[root].childlist[maxindex]
        
        # === Expand and Evaluate ===
        vlist = []
        rec_states = []
        sumv = 0

        for next_state in next_state_list:
            if self.is_end(next_state):
                v = 1.0
                vlist.append(v)
                print(f"        [EVAL] {next_state} → is_end=True → V=1.0")
            elif depth == self.max_search_depth:
                if next_state in self.NN_value:
                    v = self.NN_value[next_state]
                else:
                    v = self.gen_value(next_state)
                    self.NN_value[next_state] = v
                if v == -1:
                    sumv = -1
                    break
                vlist.append(v)
                print(f"        [EVAL] {next_state} → Value Net → V={v:.3f}")
            else:
                rec_states.append(next_state)

        if sumv != -1:
            for next_state in rec_states:
                self._dfs(next_state, depth+1, ancestors^{root})
            for next_state in rec_states:
                vlist.append(self.search_space[next_state].V)
            sumv = self._gen_V(vlist)
            print(f"        [BACKUP] Child V list: {[f'{v:.3f}' for v in vlist]} → Parent V = {sumv:.3f}")
        # ===========================

        # === Backup ===
        node = self.search_space[root]
        node.V = sumv
        node.Nlist[maxindex] += 1
        old_q = node.qlist[maxindex]
        node.qlist[maxindex] = self._gen_Q(old_q, node.Nlist[maxindex], sumv)
        new_q = node.qlist[maxindex]

        print(f"      [BACKUP] Selected Branch {maxindex} → N+1, V={sumv:.3f}, Q: {old_q:.3f} → {new_q:.3f}")
        # ==============

        return
