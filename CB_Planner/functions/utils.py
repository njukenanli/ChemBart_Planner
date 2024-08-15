from rdkit import Chem

class utils():
    @staticmethod
    def canonize(smi, allow_error = False):
        smilist = smi.split(".")
        newlist = []
        for i in smilist:
            if i == "":
                continue
            mol = Chem.MolFromSmiles(i)
            if mol is None:
                if allow_error:
                    continue
                else:
                    return None
            else:
                part = Chem.MolToSmiles(mol,canonical=True, kekuleSmiles=False)
                newlist.append(part)
        if not newlist:
            return None
        return ".".join(newlist)

    @staticmethod
    def weak_compare(str1, str2):
        l1 = str1.split(".")
        l2 = str2.split(".")
        for i in l1:
            for j in l2:
                if i == j:
                    return True
        return False
