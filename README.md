# Transformer-CNN summary


The is the 6th revised version of initial https://github.com/bigchem/transformer-cnn 
presented in those pre-print / paper:
https://arxiv.org/abs/1911.06603 & https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00423-w

The code is adapted for terminal used within OCHEM and I add few protocols to run terminal call inside a jupyter notebook via instructions.


Data Format: a csv file with one columns as SMILES or smiles and the targets (can be multiple targets and with different type ie classes and regression tasks) can be sparse too.


# Transformer-CNN Training Guide


This code was tested a batchsize 256 (use 64 for small datasets) on a RTX4090. 
It tasks 15 Go on the GPU for a run with 95k entries x n = 20.

---

## **Installation and dependencies**

### dependencies
python 3.11 (via conda see installation)
pandas
tensorflow==2.15.1
rdkit==2023.09.3
numpy==1.26.4
scikit-learn==1.3.2

### Installation:
```ini
conda create --name TCNN python=3.11
conda activate TCNN
pip install . 
```

## **Running CV Models**
In the installation folder run:

### **Step 1: Augment the Dataset**
```bash
python transformer_cnn/augment_smiles.py -i fpk.csv -o transformer_cnn/augmentedFPK.csv -n 10 -s True -t True
```
### **Step 2: Train the Model**
```bash
python transformer_cnn/runcv.py  --data transformer_cnn/augmentedFPK.csv  --output cv_results --naug 10
```

### **Step 3: Scoring report**
```bash
python transformer_cnn/scoringregcv.py -d cv_results -t FPK -n 10     
```


## **Configuration Options**
Options are available in the `config.cfg` file under the `[Details]` section:

```ini
first-line = True             # Consider dataset with one line
n_epochs = 25
batch_size = 64
early-stopping = 0.9
learning_rate = 1.0E-4
chirality = True              # Use chiral SMILES
retrain = False               # Full retrain of the embedding transformer
fixed-learning-rate = True    # Learning plateau decay
canonize = False
gpu = 0                       # Define the GPU card number to use for the run
random = True
seed = 10666                  # Default seed in OCHEM for reproducibility
augment = 20
lossmae = False               # Change the loss to MAE
```
**Suggestions:**
- `lossmae can help for dataset with visible outliers.


## **Retraining the Embedding Transformer**
To **fully retrain the embedding transformer** (learning to transform random SMILES into canonical SMILES), set:

```ini
retrain = True
```

Additionally, prepare a dataset with two columns, where each molecule's first line is structured as canonical,canonical other are randomsmiles,canonical:

```
canonicalsmile,canonicalsmile
randomsmiles,canonicalsmile
randomsmiles,canonicalsmile
...
```

---

## **Running a Single Model**
If you already have a **split dataset**, follow these steps:

### **Step 1: Augment the Dataset**
```bash
python augment_smiles.py -i train.csv -o trainaug.csv -n 10 -s True -t True
python augment_smiles.py -i apply.csv -o applynaug.csv -n 10 -s True -t True
```

### **Step 2: Train the Model**
```bash
python run.py --mode train --data trainaug.csv
```

### **Step 3: Apply / Inference**
```bash
python run.py --mode apply --data applynaug.csv
```

### **Step 4: Ensembling Augmented Predictions**
Augmentations are grouped using the `"augid"` from `df.index` with a modulo operator on the fly.  
To ensure proper functioning, you **must pass the correct `-n` augmentation number**.

---

### **License**
MIT License


