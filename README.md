
The is a revised version from https://github.com/bigchem/transformer-cnn 

https://arxiv.org/abs/1911.06603 & https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00423-w

Code was adapted to be used within OCHEM and few extensions have be made to also use it from jupyter notebook via 
commandline instructions.


In jupyter for a CV of 5 for example you will need to split your data into 10 files (5 for train , 5 for test)

Internal validation is done in the training method

in commandline this is done like this 

step 1: augment
python augment_smiles.py  -i train.csv -o trainaug.csv -n 10 -s True -t True
python augment_smiles.py  -i apply.csv -o applynaug.csv -n 10 -s True -t True

step 2: training
python run.py --mode train --data trainaug.csv

step 3: apply
python run.py --mode apply --data applynaug.csv

step 4: ensembling augmentated prediction

Apply mode generate a results file that you need to average based on number of augmentation 
augmentation is made by row extensions so one row become N rows in the dataset. 

You can group by using the "augid" from the applynaug file.py

python augment_smiles.py  -i apply.csv -o apply1naugidx.csv -n 10 -s True -t False

python scoringreg.py -o results.csv -i applynaug.csv -n Result0

