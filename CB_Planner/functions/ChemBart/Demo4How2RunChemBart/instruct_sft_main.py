'''
Run the code using DDP with 4 GPUs:

python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 main.py
'''

'''
input: [
    "<cls>original_molecule_1><n00>>optimized_molecule_2<end>", # First property
    "<cls>original_molecule_2><n00>>optimized_molecule_2<end>", # First property
    "<cls>original_molecule_3><n01>>optimized_molecule_3<end>", # Second property
    "<cls>original_molecule_4><n02>>optimized_molecule_4<end>", # Third property
]


'''

import ChemBart
import os
import json
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
c=ChemBart.ChemBart()
if True:
    with open("complete_train_string.json") as f:
        stringlist=json.load(f)
else:
    stringlist=["<cls>CC=CC=C><n01>>CC(Br)C=CC(Br)<end>"]
c.instruct_sft_parallel(stringlist, batch_size = 2)

'''
For inference:
c.predict("<cls>original_mol><task_token>><msk><end>", 
            decoder_input = "<cls>original_mol><task_token>>",
            top_k=10, max_len=200, stop_with_sep = True)
'''

