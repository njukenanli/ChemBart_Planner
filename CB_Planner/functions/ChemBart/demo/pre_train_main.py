import ChemBart
from argparse import ArgumentParser
import json

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--minibatch", type=int, default=4)
    parser.add_argument("--accumsteps", type=int, default=8)
    parser.add_argument("--ckptpath", type=str, default="./t5pretrainckpt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--greedy_minibatch", type=int, default=40)
    parser.add_argument("--beamsearch_minibatch", type=int, default=1)
    args = parser.parse_args()

    if not args.debug:
        with open("pre-train-dataset/complete_mit_train.json") as f:
            trainset=json.load(f)
        with open("pre-train-dataset/complete_mit_test.json") as f:
            testset=json.load(f)
    else:
        trainset=[
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
        ]
        testset=trainset
    c=ChemBart.ChemBart(None, "cpu")
    c.pretrain(
        trainset,
        testset,
        args.minibatch,
        args.accumsteps,
        1e-5,
        100,
        768,
        "log-t5-pretrain.txt",
        args.ckptpath,
        True,
        args.greedy_minibatch,
        args.beamsearch_minibatch,
    )
