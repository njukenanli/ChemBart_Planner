TRAIN = True
from MCTS_PV_train.model4mcts import BatchCB_MCTS
import json

with open("data/filtered_MCTS_data.json") as f:
    l = json.load(f)

c = BatchCB_MCTS("model/CB_MCTS_FULL4_batch.pth","model/ChemBart_FULL_4.pth","cuda:0")
if TRAIN:
    train_size = int(len(l)*5/6)
    valid_size = int(len(l)*1/12)
    test_size = len(l)-train_size-valid_size
    c.batch_train(data = l, epochs = 200, tr = train_size, val = valid_size, te = test_size, batch_size=1)
else:
    c.test(l[int(len(l)*11/12):])

