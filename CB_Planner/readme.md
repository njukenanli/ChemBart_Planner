# Technical Document for Multi-step Planning with MCTS and ChemBart (V2.0) #

Kenan Li

## 1. Synthetic Route Design Overview ##
Two steps are involved to give a complete synthetic route:

1. Use the MCTS and RL architecture to give a pure synthetic route without reaction information (conditions etc.) expressed in the form of tree structure. Note that reagents of each step would also be generated in the process of precursor prediction.

2. Use the fine-tuned models to add other reaction information such as temperature, yield and so on to each step. 

## 2. Multi-step Planning with Reinforcement Learning ##
In this section we only talk about designing a single (the best) route for a target molecule; while the technique for designing alternative routes for a single molecule will be talked in **Section 4**.

Mostly we follow the MCTS algorithm described in the ReSynZ paper (Citation: J. Chem. Theory Comput. 2024, 20, 11, 4921 - 4938. URL: [https://pubs.acs.org/doi/10.1021/acs.jctc.4c00071?ref=pdf](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00071?ref=pdf)). The MCTS library developed by Kenan Li is adopted as the source code of MCTS in this program ([https://github.com/njukenanli/MCTS-python-multiprocessing](https://github.com/njukenanli/MCTS-python-multiprocessing)). To understand how to use it, see the github website shown above. The major difference between the original ReSynZ multi-step algorithm (template-based method) and the new one is that the number of child nodes for any father node (a molecule) in the template-based program is fixed, which is equal to the total number of rules in the rule set, whereas the number of child nodes for a single node in our new program is determined by the number of different precursors that our language model actually generates.

To generate the “best” route for a target molecule, in each step, we greedily pick the child node with the maximum probability in the  output of MCTS, that is, the probability distribution for all the child nodes of the father node in this step. In order to drive the MCTS algorithm to get the  for a node, we need to provide the functions to generate all the child nodes (possible synthesis precursors) for a father node (a product molecule), as well as calculate the policy function and value function for this node. In this version we design two functions to satisfy such requirements.

**The first function (CB\_Planner.functions.nn.GenChildnodePolicy)** is to generate the child nodes for a father node and get the policy for these child nodes in the meantime. Firstly we input the canonized SMILES expression of the father node into the pre-trained model (loaded on GPU0) and yield top-k answers along with their probability distribution p<sub>0</sub> = (p<sub>1</sub>’, p<sub>2</sub>’, ...). The probability of each child node here describes the occurrence frequency of its similar cases in the literature. Then we use the rdkit module to canonize the SMILES expressions of all the child nodes

`Chem.MolToSmiles(Chem.MolFromSmiles(smiles_string),canonical=True, kekuleSmiles=False)`

and filter those which cannot be canonized because they are grammatically wrong expressions (generated due to the hallucination of LLM). Subsequently, we deduplicate the child nodes which have the same canonized SMILES expressions. We only retain one of all the child nodes with the same expressions and add up all their probabilities as the new probability for the retained child node. We also remove the child nodes which contain the same molecules as their father nodes. Drawing on TTL, we also need to filter those unreliable precursors predicted, due to the hallucination problem in LLM, by using the pre-trained model (this copy is loaded on GPU1). To be specific, we need to input each predicted precursor with its corresponding product (the father node) into the pre-trained model to generate their possible reagents, and then input into the model each predicted precursor and its corresponding reagents to generate their corresponding product. If the predicted product, after canonization, is not the same with the original product (the father node), then the child node corresponding to this precursor would be discarded. We collect the child nodes left along with their corresponding probabilities p<sub>1</sub> = (p<sub>11</sub>, p<sub>12</sub>, ...). After that, we use the fine-tuned model for MCTS data regression (loaded on GPU2) to get the policy function p<sub>2</sub> = (p<sub>21</sub>, p<sub>22</sub>, ...) for these retained child nodes. The probability of each node in the policy function means the ease with which the father node could be retro-synthesized into basic molecules by going across this child node. (The higher, the better.) In order to integrate these two experiences from literature and exploration, we normalize p<sub>1</sub> and p<sub>2</sub>, time the corresponding item to get p<sub>3</sub> = (p<sub>11</sub>\*p<sub>21</sub>, p<sub>12</sub>\*p<sub>22</sub>, ...), and normalize P<sub>3</sub> to generate the final policy vector to be input into the MCTS algorithm. From another perspective, P<sub>3</sub> integrates the experiences from supervised learning and reinforcement learning, which is now the popular trend in those famous CASP works like PDVN, Retro*+ and so on. 

**The second function (CB_Planner.functions.nn.Value)** is to calculate the value function for each node. We simply input into the fined-tuned model the canonized SMILES expression and get its value function. If there are still abundant time left for us, the MCTS results from this program can also be collected to further train the fine-tuned model for MCTS data prediction.

The structure of the MCTS module in this program is shown in the following figure.

![](../img/img1.png)

<sub>Annotation</sub>
<sub>RL\_Policy, RL\_Value: the functions to generate the policy vector and the value respectively with the fine-tuned ChemBart for MCTS data regression</sub>


## 3. Prallelism ##
We do not plan to adopt parallelism in the MCTS algorithm. That’s because the GPU memory shortage in LLM results in that only few functions can call the same model on a GPU in the meantime (1 function for 11GB GPU, and 4 for 24GB GPU, for our model). Therefore, if multiple threads or processes from different branches in a tree call a GPU at the same time, they need to queue to wait. As a result, there is no significant acceleration if parallelism is adopted. What’s more, the cost to create threads or processes would add up to the time of MCTS. Even worse, the Python interpreter only implements process parallelism, the cost of which is much higher than threads. This means parallelism among branches in our program may even make MCTS slower! **This is why we do not use parallelism in DFS.**

However, if the memory of our GPU is more than 20GB which allows multiple processes to call a LLM on the GPU simultaneously, then it is suggested that if there are several molecules to be analyzed, each molecule could be allocated with one process. In this way, we only create multiple processes at the beginning, and limited and fixed number of processes are created, while much more processes would be created and deleted dynamically if we adopt parallelism among different branches of the search tree, where both the larger number of processes and the process of creating and deleting processes would contribute to the computing cost.

**It is suggested that the “multiprocessing” module in Python be adopted to allocate one MCTS process to each target molecule to be analyzed, and the “semaphore” process lock be used to limit the number of processes that are able to call a GPU at the same time.** For 11GB GPU, the semaphore is set to 1, which is the same as a “mutex” lock, while for 24GB GPU, the semaphore is estimated to be up to 3. Also, in process parallelism, each process would generate some memory usage in each GPU due to the “share memory” technique of the PyTorch model. So we need to use the Pool in multiprocessing to limit the number of processes that are running at the same time in case of OOM or just seizure.

## 4. Alternative Routes ##
In **Section2**, we talked about how to generate the best synthesis route for a molecule, which chooses the child node with the highest probability in each step. For alternative routes, in ReSynZ, one of the child nodes with the top-5 probabilities is randomly chosen in each step, but such scheme is too simple and the restriction is too loose, as it may always choose the same child nodes in different routes designed or frequently choose child nodes with low probabilities like the 4th or 5th ones in a route designed.

It is suggested to use **beam search** to generate alternative answers. **If we need to generate k routes, we choose the k child nodes with top-k probabilities in a step for each route designed to get k2 routes, then select the k routes from the k2 routes with the highest total probabilities, where the total probability of a route is the geometric mean of probabilities in all of its steps.** Then go to the next step and loop the process above. In this way, we ensure that the k routes generated are different from each other and are the top-k best ones among all possible routes. 


Version 1.0: 2024.7.28

Version 2.0: 2024.8.27