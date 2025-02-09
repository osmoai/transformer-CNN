License MIT



The is the 6th revised version of initial https://github.com/bigchem/transformer-cnn 
presented in those pre-print / paper:
https://arxiv.org/abs/1911.06603 & https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00423-w

The code is adapted for terminal used within OCHEM and I add few protocols to run terminal call inside a jupyter notebook via instructions.

In jupyter for a CV of 5 for example you will need to split your data into 10 files (5 for train , 5 for test)

Internal validation is done in the training method

Data Format: a csv file with one columns as SMILES or smiles and the targets (can be multiple targets and with different type ie classes and regression tasks) can be sparse too.

Mini Guide: 

Dependencies:
python 3.11 (via conda see installation)
pandas
tensorflow==2.15.1
rdkit==2023.09.3
numpy==1.26.4
scikit-learn==1.3.2

Installation:
to get a local instance 
conda create --name TCNN python=3.11
conda activate TCNN
pip install . 


Running 5CV model: 
to run a CV model first prepare your dataset

python transformer_cnn/augment_smiles.py -i fpk.csv -o transformer_cnn/augmentedFPK.csv -n 10 -s True -t True

Then run the models (5 by defaults)

python transformer_cnn/runcv.py  --data transformer_cnn/augmentedFPK.csv  --output cv_results --naug 10

Get the report:
python transformer_cnn/scoringregcv.py -d cv_results -t FPK -n 10     


this code was tested a batchsize 256 (use 64 for small datasets) on a RTX4090. 
It tasks 15 Go on the GPU for a run with 95k entries x n = 20.

Options list available via config.cfg file [Details] part:
first-line = True # consider dataset with one line
n_epochs = 25
batch_size = 64
early-stopping = 0.9
learning_rate = 1.0E-4
chirality = True # use chiral smiles 
retrain = False # full retrain of the embedding transformer
fixed-learning-rate = True # learning plateau decay
canonize = False 
gpu = 0 # define the gpu card number to use for the run. 
random = True
seed = 10666 # default seed in OCHEM for all models for reproductibility
augment = 20
lossmae = False # change the loss to MAE

Suggesting to run a retrain can help, lossmae can help for dataset with visible outliers.

Important:
To retrain completely the embedding transformer (learn to make the random smiles to canonical smiles transformation), you have to set parameter "retrain" to True and prepare a dataset with two columns, with first line per molecule is always "canonical,canonical"
like this:

randonsmiles,canonicalsmile


Remark the single run:


Running a single model with data already splited:
step 1: augment the dataset
python augment_smiles.py  -i train.csv -o trainaug.csv -n 10 -s True -t True
python augment_smiles.py  -i apply.csv -o applynaug.csv -n 10 -s True -t True

step 2: training 
python run.py --mode train --data trainaug.csv

step 3: apply / inference
python run.py --mode apply --data applynaug.csv

step 4: ensembling augmentated prediction 
We group augmentation using the "augid" from the df.index modulo operator on the fly. To work properly you must pass the correct -n argument number of augmentation.




