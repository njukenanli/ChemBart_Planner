config = \
{
            # params for supervision
            "getdata": {
                "train": False,  # retain search data for RL training or not
                "max_train_data_num": 5000,  # if retain training data, at most how many data should be retained.
            },

            # params for computing
            "mcts": {
                "mcts_times": 50, # iteration time per step
                "max_route_len": 16, # the max length of the route designed, if the program exceeds the limit but still haven't finished, return failed.
                "max_search_depth": 8, # the max depth of DFS in MCTS
                "temp_coef" : 1.0, # the temperature coefficient of policy 
                "Cpuct" : 1.0, # the Cpuct coeeficient in mcts
                "update_method" : 'avg', # or 'min'
                "debug": False, # print informantion when MCTS
            },

            "nn":{
                "gen_dev": "cuda:0", # device to load generation model (pre-trained model)
                "val_dev": "cuda:1", # device to load verification model (pre-trained model)
                "rl_dev": "cuda:2", # device to load RL model (fine-tuned model)
                "temp_yield_dev": "cuda:0", # device to load model for temperature and yield prediction (fine-tuned model)
                "choiceperstep": 10, # num of choices generated in each node in the tree.
            },

            # params for computing resources. If the cuda returns error or the gpu just does not work, reduce this params!
            "parallel":{
                "process_parallel": True,
                "pool_size": 5, # num of molecules (processes) that can be solved at the same time. Each process requires one cpu (so do not exceed num of cpus) and part of the gpu memory (so usually this param is much smaller than num of cpus).
                "semaphore_per_model": 2, # num of processes that can call a model at the same time, linited by the gpu memory.
            }
}

