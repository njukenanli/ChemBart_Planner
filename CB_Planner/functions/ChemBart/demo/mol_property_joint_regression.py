from ChemBart import CB_multi_task_sep_regression
import json

c = CB_multi_task_sep_regression(name = "mol_property",
                                num_feature = 5,
                                binary_classification = True, 
                                device = "cuda:0")
with open("your_data.json") as f:
    data = json.load(f)


'''

Your input should be: ["<cls>molecule_smiles<task_token><end>", label]
where <task_token> is <n00> <n01> <n02> <n03> <n04> representing different properties, 
and label is float number

Let's stipulate:

To predict BBBP, input "<cls>molecule_smiles<n00><end>"
To predict HIV, input "<cls>molecule_smiles<n01><end>"
To predict BACE, input "<cls>molecule_smiles<n02><end>"
To predict Tox21, input "<cls>molecule_smiles<n03><end>"
To predict ClinTox, input "<cls>molecule_smiles<n04><end>"

'''


c.single_train(data, 
               epoch = 50, 
               tr = 16, # The first tr samples are training data
               val = 2, # The [tr: tr + val] samples should be validation data
               grad_accumulate = 4, # 4 sample one gradiaent update
               )

'''
To predict a result:
c.pred_one_instance("<cls>molecule_smiles<task_token><end>")
For example, 
To predict ClinTox,
c.pred_one_instance("<cls>molecule_smiles<n04><end>")
'''