config = \
{
            "save_path": './answer' ,
            "bascic_mol": "./CB_Planner/functions/ChemBart/data/basic_mol.json",
            "chembart_path":'./CB_Planner/functions/ChemBart/model/Pretrained-Full.pth', 
            # please use ChemBart pretrained on USPTO-FULL! We find though MIT dataset has better data quality but its smaller data size (1/3 of USPTO-full) makes generalizability very terrible and thus it's not suitable for large-scale out-of-scope prediction.
            "rl_path":'./CB_Planner/functions/ChemBart/model/Policy-Value.pth',
            "ty_path":'./CB_Planner/functions/ChemBart/model/Temperature-Yield.pth', 
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
                "gen_dev": "cuda:0", # device to load retrosynthesis model (pre-trained model)
                "agt_dev": "cuda:1", # device to load reagent model (pre-trained model)
                "pro_dev": "cuda:2", # device to load forward (round trip) model (pre-trained model)
                "rl_dev": "cuda:3", # device to load RL model (fine-tuned model)
                "temp_yield_dev": "cuda:0", # device to load model for temperature and yield prediction (fine-tuned model)
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



