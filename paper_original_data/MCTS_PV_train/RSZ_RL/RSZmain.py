
TRAIN = True
from RetroNNet import RetroNNet
import torch
from procdata import RSZ_processor
data_path = '../ChemBart/data/filtered_MCTS_data.json'
proc = RSZ_processor(data_path)
l = proc.process()
#model = RetroNNet(2048, 11950)
model = RetroNNet(2048, 13312)
if TRAIN:
    train_size = int(len(l)*5/6)
    valid_size = int(len(l)*1/12)
    test_size = len(l)-train_size-valid_size
    model.fit(l, train_size, valid_size, test_size, batch_size=4098, dev = "cuda:0")
else:
    out = model.test(l[11000:], dev = "cuda:0")
    print(out)
