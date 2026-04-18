# ChemBart Planner: Retrosynthesis Route Planning Program with a Pre-trained LLM #

Paper: Coming ...

This is the multi-step synthesis route planner based on our pre-trained LLM: ChemBart. In this program, we use our pre-trained and fine-tuned ChemBart models to generate synthesis precursors for target product molecules, and reaction information for each step like reagents, temperature, yield and so on. We use the MCTS and RL algorithms in our previous work ReSynZ([https://pubs.acs.org/doi/10.1021/acs.jctc.4c00071?ref=pdf](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00071?ref=pdf)) for the selection of precursors in each synthesis step.

## Preparation ##

### Enviroment ###

System: Linux.

Stable version: main branch

```bash
git clone https://github.com/njukenanli/ChemBART_Planner
```

We recommend using conda to set up the environment. The environment.yml file is provided in the repository.

```bash
conda env create -f environment.yml
conda activate myproject
```

Typical installation time on a Linux machine: 10 min.

### Data ###

Datasets should also be downloaded separately to run the planner. 

Model weights and basic molecule dataset have been uploaded to https://huggingface.co/ChemBart

We don't provide model weights for molecular property regression but you can reproduce the results following our paper.

To download necessary model weights and dataset to run ChemBart Retrosynthesis Planner:

```bash
python -m CB_Planner.data.download --timeout 600
```

## Reference ##

For ChemBart source code, see: [CB_Planner/functions/ChemBart/](https://github.com/njukenanli/ChemBart_Planner/tree/main/CB_Planner/functions/ChemBart).

For demo on how to quickly run single step inference using ChemBart model for synthesis, temperature&yield, policy&value, refer to [CB_Planner/functions/ChemBart/demo/single_step_demo.ipynb](CB_Planner/functions/ChemBart/demo/single_step_demo.ipynb). Prediction of one sample typically takes around 1 min.

For implementation details of this planner, see: [CB_Planner/](https://github.com/njukenanli/ChemBart_Planner/tree/main/CB_Planner)

Below we provide manual on multi-step retrosynthesis.

## Planning ##
1.Target Molecules

Put the target molecules you want to analyse in the tasklist.py file. Each molecule should be written in the form (SMILES expression, number of alternative routes you want to get). Usually the number of alternative routes is set to be around 1~3.

2.Config

Edit the parameters in the config.py file to control the planner. The discriptions of the parameters are also in the file. Here are some suggested values for some of the parameters.
    
    - mcts_times: 50 - 100 # DFS iteration time per step. As the running of LLM is usually very slow, the number of DFS per step should not be set too high.
    - max_route_len: 12 - 20, # The max length of the route designed, if the program exceeds the limit but still haven't finished, return failed. If this paramater is set too low, many routes would end up uncompleted. On the contrary, It is also unnecessary to set this parameter too high. Usually most synthesis programms cannot handle routes longer than 20.
    - max_search_depth: 8, # The max depth of DFS in MCTS. Usually the higher the better, but also much slower.
    - choice_per_step: 10, # Usually LLM can only generate around 10 choices with high quality. We need to make more efforts to generate more choices in each step to increase the design success rate.
    - device: # There are currently 3 models to be loaded simultaniously. If the memory of each GPU is less than 32 GB, each GPU can only load one model. If there are less than 3 GPUs and the memory of each GPU is less than 32 GB, put the rest of the models on CPUs.
    - pool_size: # If there are more than one cpus, process parallel here can be considered. If users use GPUs, each process would generate some memory usage in each GPU due to the “share memory” technique of the PyTorch model, so the Pool is needed in multiprocessing to limit the number of processes that are running at the same time in case of OOM or just seizure. For 11GB GPU, pool size should be no larger than 3; for 24GB, 6; for 48GB, 12.
    - semaphore_per_model: # num of processes that can call a model at the same time, limited by the gpu memory. For 11GB GPU, 1; for 24GB, no larger than 3; for 48 GB, 6.
    

3.Run the Planner

Run the main.py to start route planning.

>     python main.py

For background running,

>     nohup python main.py > log.out 2>&1 &


## Results ##
The computation results are in the directory "answer". The name of each result file is ‘answer \_ target molecule index \_ target molecule SMILES \_ route \_ alternative route index for this molecule’ and each file shows a route. In each file, it first shows whether the route design is successful and the total probability of the route (the product of the probabilities of each step); then the synthesis route, which is expanded in the format of ‘target molecule: {its  precursors respectively}’; and finally, the information of each step, including the reagents, the temperature, and the yield, which are predicted by our pre-trained and fine-tuned model.

If you set to retain MCTS process data for RL in config.py, the train data would also be saved here. The format is "[[product molecule, precursor list], [value, policy]]".

If you would like to process the results after MCTS, add codes in main.py after 

>     ans = planner.plan(tasklist)

or in CB\_Planner/board.py after

>     ty.AddTemperatureYield(routes)
>     del ty



