# Machine Learning for semileptonic ttbar

# Step 0: Setup environment
FIRST TIME SETUP ONLY

Clone this repository.
```
git clone https://github.com/hohare/ttbar_ml.git
```

Initialize the submodule.
```
git submodule init
git submodule update
```

Create virtual environment.
```
micromamba env create -f environment.yml
```

# Step 0.1: Use environment
After first time setup has been completed, you need to activate the virtual environment using micromamba.
```
source activate_env.sh
```
# Step 1: Extract information from root files (pretraining)
Running a coffea processor requires having a python dictionary with the dataset name as the key and the list of .root files as the value. The script does not currently handle more than one dataset in the dictionary (will put everything into a single object regardless of dataset).

Pretraining varies slightly for the projects supported by this repository.
Currently, the dctr project outputs training features and non-training variables into a pandas dataframe whereas the sbi project directly concatenates pytorch tensors. The goal is to eventually have both output pytorch tensors (after we are confident in the ability of our selected features to perform up to standards).

Example command to run over an eft sample for the dctr project:
```
python run_pretraining_processor.py Inputs/pretraining/TT01j1l_modCentral.json -w Inputs/rwgt_card_modCentral.dat -u dctr
```

# Step 2: Train a neural network (training)
The settings for training a neural network should be organized into a .yml file. See Inputs/training/README.md for details.

Example command to train for the dctr project:
```
python run_training.py Inputs/training/modCentral_noWgtNorm.yml dctr
```

# Step 3: Use the information to do something (projects)
Currently this repository has 2 projects that use the output of a neural network.
1. dctr: Uses a network trained on GenParts top particles to reweight events from a MadGraph LO to Powheg NLO.
2. sbi: Uses a network trained on ttbar decay product momentum 4-vectors to reweight to arbitrary points in EFT phase space and set limits.

