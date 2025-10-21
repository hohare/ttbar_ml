import os
import gzip
import cloudpickle
import json
import yaml
import numpy as np
import pandas as pd
from torch import from_numpy, save, cat, tensor, float64, where, ones_like, manual_seed, initial_seed
from torch.utils.data import TensorDataset, random_split

from training.normalizations import apply_weight_norm, apply_feature_norm

with open('Inputs/xsec_semilep.json') as f:
    xsecs = json.load(f)
with open('ttbar_utilities/inputs/luminosity_in_fb.json') as f:
    lumis = json.load(f)
    
def convert_to_net_tensors(config, dataframe, isEFT=False, doWgtNorm=True):
    # Extract features
    features = from_numpy(dataframe[config['features']].values).to(dtype=float64)
    nonfeat_list = [k for k in dataframe.columns if k not in config['features']]
    nonfeatures = from_numpy(dataframe[nonfeat_list].values).to(dtype=float64)
    # Handle if is eft or is powheg
    if isEFT: 
        useWeights = 'SMweights'
        labels = tensor([1], dtype=float64).repeat(features.size(0))
    else: 
        useWeights = 'genweights'
        labels = tensor([0], dtype=float64).repeat(features.size(0))
    # Extract weights
    wgts = from_numpy(dataframe[useWeights].values).to(dtype=float64)

    return TensorDataset(features, wgts, labels), nonfeatures

def prep_dataset(dataset, config, validation=False, masking=True, verbose=False):
    # Features
    umm=True
    if umm:
        if 'fmeans' in config:
            if verbose: print('Normalizing features by provided')
            features, _, _ = apply_feature_norm(dataset[:][0], 
                                                tensor(config['fmeans']),
                                                tensor(config['fstds']))
        else:
            if verbose: print('Normalizing features')
            features, fmeans, fstds = apply_feature_norm(dataset[:][0])
            config['fmeans'] = fmeans.tolist()[0]
            config['fstds']  = fstds.tolist()[0]
    else:
        features=dataset[:][0]
    
    if validation:
        return features
        
    # Weights
    if "sow" in config:
        sig_sow = config['sow']
        sig_sow = config['sow']
    else:
        sig_sow  = dataset[:][1][dataset[:][2]==1.].sum()
        bkg_sow  = dataset[:][1][dataset[:][2]==0.].sum()

    #xsec normalization
    sig_norm = xsecs['UL2017']['modCentral'] / sig_sow * lumis['UL2017']*1000.
    bkg_norm = xsecs['UL2017']['powheg']     / bkg_sow * lumis['UL2017']*1000.
    wgts = where(dataset[:][2]==1., dataset[:][1]*sig_norm, dataset[:][1]*bkg_norm)
    #weight normalization
    if not validation and config['normalization']=='weightMean':
        if verbose: print('Normalizing weights by weightMean')
        sig_wgt_mean = wgts[dataset[:][2]==1.].mean()
        bkg_wgt_mean = wgts[dataset[:][2]==0.].mean()
        config['test_wgtmeans'] = [bkg_wgt_mean.tolist(), sig_wgt_mean.tolist()]
        wgts = where(dataset[:][2]==1., wgts/sig_wgt_mean, wgts/bkg_wgt_mean)

    # Mask weights and mass
    if masking:
        # Remove negative weights and
        # remove events with top mass outside [152., 193.]
        lower_mask = (dataset[:][0][:,3]>=152.) & (dataset[:][0][:,7]>=152.)
        upper_mask = (dataset[:][0][:,3]<=193.) & (dataset[:][0][:,7]<=193.)
        mask = (wgts>=0.) & lower_mask & upper_mask
    else:
        mask = (wgts != np.inf)
        
    return TensorDataset(features[mask], wgts[mask], dataset[:][2][mask]), config
    

def create_train_dataset(config, validation=False):
    name = config['name']
    os.makedirs(name, mode=0o755, exist_ok=True)
    if config['normalization'] is None: doWgtNorm = False
    elif config['normalization']==1: doWgtNorm = False
    else: doWgtNorm = True
    config['seed'] = initial_seed()
    # Convert dataframes to tensor
    sig_df = []
    for i in config['signalSample']:
        with gzip.open(i, 'rb') as f:
            sig_df.append(cloudpickle.load(f))
    sig_df = pd.concat(sig_df, axis=0)
    bkg_df = []
    for i in config['backgroundSample']:
        with gzip.open(i, 'rb') as f:
            bkg_df.append(cloudpickle.load(f))
    bkg_df = pd.concat(bkg_df, axis=0)
    
    # Cap number of events in train/test set
    cap = np.min([int(3.1e6), sig_df.shape[0], bkg_df.shape[0]])
    print('Keeping {:.2E} events per dataset'.format(cap))
    # Shuffle and Convert pandas dataframes to pytorch tensors
    sig_dataset, sig_non_features = convert_to_net_tensors(config, sig_df.sample(cap), isEFT=True)
    bkg_dataset, bkg_non_features = convert_to_net_tensors(config, bkg_df.sample(cap), isEFT=False)
    
    # Combine datasets into single TensorDataset
    dataset = TensorDataset(
        cat([bkg_dataset[:][0], sig_dataset[:][0]]),
        cat([bkg_dataset[:][1], sig_dataset[:][1]]),
        cat([bkg_dataset[:][2], sig_dataset[:][2]]),
    )
        
    # Split dataset into training and testing
    train, test = random_split(dataset, config['split'])

    # feature and weight normalization
    train, config = prep_dataset(train, config)
    test, config  = prep_dataset(test, config)
    
    # Contruct the datasets
    config['traindata'] = f'{name}/train_tensor.pkl'
    print('Training sample has {:.2E} events'.format(train[:][1].shape[0]))
    save(train, config['traindata'])
    
    config['testdata'] = f'{name}/test_tensor.pkl'
    print('Testing sample has {:.2E} events'.format(test[:][1].shape[0]))
    save(test, config['testdata'])

    del train, test, dataset
    # Update config
    with open(config['name']+'/training.yml', 'w') as f:
        yaml.dump(config, f)

    if validation:
        dataset = TensorDataset(
            cat([bkg_non_features, sig_non_features]),
            cat([bkg_dataset[:][1], sig_dataset[:][1]]),
            cat([bkg_dataset[:][2], sig_dataset[:][2]]),
        )
        _, test = random_split(dataset, [0.8, 0.2])
        test, config  = prep_dataset(test, config, masking=False)
        config['variabledata'] = f'{name}/var_tensor.pkl'
        print('Var sample has {:.2E} events'.format(test[:][1].shape[0]))
        save(test, config['variabledata'])
        
    return config