import ChemBart
from argparse import ArgumentParser
import json

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--minibatch", type=int, default=4)
    parser.add_argument("--accumsteps", type=int, default=8)
    parser.add_argument("--ckptpath", type=str, default="./pretrainckpt")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    c=ChemBart.ChemBart(None, "cpu")
    if not args.debug:
        with open("complete_train_string.json") as f:
            trainset=json.load(f)
        with open("complete_test_string.json") as f:
            testset=json.load(f)
    else:
        trainset=[
            "CC=CC=C.BrBr>C(Cl)(Cl)(Cl)(Cl)>CC(Br)C=CC(Br)",
            "F.O=C(OCc1ccccc1)N1CCC2(CC1)CO2>O=C([O-])O.ClCCl.[Na+].c1ccncc1>O=C(OCc1ccccc1)N1CCC(F)(CO)CC1",
            "O=C(O)Cc1cccc([N+](=O)[O-])c1>B.C1CCOC1.CO>O=[N+]([O-])c1cccc(CCO)c1",
            "N#Cc1ccc([N+](=O)[O-])cc1C#N.Oc1cccnc1>[K+].[K+].O=C([O-])[O-].CN(C)C=O>N#Cc1ccc(Oc2cccnc2)cc1C#N",
            "CCCC(=O)C1=C(Cl)CC(C2CCCSC2)CC1=O.c1cn[nH]c1>CN(C)C=O>CCCC(=O)C1=C(n2cccn2)CC(C2CCCSC2)CC1=O",
            "O=C1OC(=O)C2=C1CCCC2.Nc1ccc(Cl)cc1F>CCOCC>O=C(O)C1=C(C(=O)Nc2ccc(Cl)cc2F)CCCC1",
            "COC(=O)C1CCCC1C1CC(c2ccc(O)cc2)=NO1.Cc1cc(CCl)c2ccccc2n1>O=C([O-])[O-].[I-].[K+].[K+].[K+]>COC(=O)C1CCCC1C1CC(c2ccc(OCc3cc(C)nc4ccccc34)cc2)=NO1",
            "CCc1cccc(C)c1CNc1cccn2c(C)c(CO)nc12.O=C1CCC(=O)O1>CC#N.[H-].[Na+]>CCc1cccc(C)c1CNc1cccn2c(C)c(COC(=O)CCC(=O)O)nc12",
            "N#CN1CCCCC1.[Na+].C[O-]>CO>COC(=N)N1CCCCC1",
            "Cc1nn(-c2ccc(CCO)cc2)c(C)c1-c1ccccc1.O=S(Cl)Cl>Cc1ccccc1.ClC(Cl)Cl>Cc1nn(-c2ccc(CCCl)cc2)c(C)c1-c1ccccc1",
        ]
        testset=trainset
    c.pretrain(
        trainset,
        testset,
        args.minibatch,
        args.accumsteps,
        1e-5,
        100,
        768,
        "log.txt",
        args.ckptpath,
    )
