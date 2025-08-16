config = \
{
            "save_path": 'JMC/benchmark_topp_strict/' ,
            "bascic_mol": "CB_Planner/data/basic_mol.json",
            "chembart_path":'/home/zhangyijian/ChemBart/v2/ChemBart_model/ChemBart_FULL_4.pth',
            "rl_path":'/home/zhangyijian/ChemBart/CB_Planner/CB_Planner/functions/ChemBart/model/CB_MCTS.pth',
            "ty_path":'/home/zhangyijian/ChemBart/CB_Planner/CB_Planner/functions/ChemBart/model/temp_yield_bart.pth', 
            # params for supervision
            "getdata": {
                "train": False,  # retain search data for RL training or not
                "max_train_data_num": 5000,  # if retain training data, at most how many data should be retained.
            },

            # params for computing
            "mcts": {
                #"mcts_times": 50, # iteration time per step
                "mcts_times": 10,
                "max_route_len": 8, # the max length of the route designed, if the program exceeds the limit but still haven't finished, return failed.
                "max_search_depth": 8, # the max depth of DFS in MCTS
                "temp_coef" : 1.0, # the temperature coefficient of policy 
                "Cpuct" : 1.0, # the Cpuct coeeficient in mcts
                "update_method" : 'avg', # or 'min'
                "debug": False, # print informantion when MCTS
            },

            "nn":{
                "gen_dev": "cuda:2", # device to load generation model (pre-trained model)
                "agt_dev": "cuda:2", # device to load agent model (pre-trained model)
                "pro_dev": "cuda:2", # device to load product (round trip) model (pre-trained model)
                "rl_dev": "cuda:2", # device to load RL model (fine-tuned model)
                "temp_yield_dev": "cuda:2", # device to load model for temperature and yield prediction (fine-tuned model)
                #"choiceperstep": 50, # num of choices generated in each node in the tree.
                "sampling_method": 'top_p', #'top_k' / 'top_p' / 'beam'
                "topk": 10,
                "topp": 0.9,
                "num_samples": 50,
                #"temperature":1.0
            },

            # params for computing resources. If the cuda returns error or the gpu just does not work, reduce this params!
            "parallel":{
                "process_parallel": False,
                "pool_size": 3, # num of molecules (processes) that can be solved at the same time. Each process requires one cpu (so do not exceed num of cpus) and part of the gpu memory (so usually this param is much smaller than num of cpus).
                "semaphore_per_model": 3, # num of processes that can call a model at the same time, linited by the gpu memory.
            }
}

