License MIT

The is the 6th revised version of initial https://github.com/bigchem/transformer-cnn 
presented in those pre-print / paper:
https://arxiv.org/abs/1911.06603 & https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00423-w

The code is adapted for terminal used within OCHEM and I add few protocols to run terminal call inside a jupyter notebook via instructions.

In jupyter for a CV of 5 for example you will need to split your data into 10 files (5 for train , 5 for test)

Internal validation is done in the training method

Use only commandline you can do like this: 

step 1: augment the dataset
python augment_smiles.py  -i train.csv -o trainaug.csv -n 10 -s True -t True
python augment_smiles.py  -i apply.csv -o applynaug.csv -n 10 -s True -t True

step 2: training 
python run.py --mode train --data trainaug.csv

step 3: apply / inference
python run.py --mode apply --data applynaug.csv

step 4: ensembling augmentated prediction
You can group by using the "augid" base on df.index modulo operator on the fly so you must pass the correct -n argument number of augmentation !

python scoringreg.py -o results.csv -i applynaug.csv -t Result0 -n 10

this code was tested a batchsize 256 (use 64 for small datasets) on a RTX4090. 
It tasks 15 Go on the GPU for a run with 95k entries x n = 20.

