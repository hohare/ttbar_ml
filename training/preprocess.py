import os
import gzip
import cloudpickle
import json
import yaml
import numpy as np
import pandas as pd
from torch import from_numpy, save, cat, tensor, float64
from torch.utils.data import TensorDataset, random_split

from training.normalizations import apply_weight_norm, apply_feature_norm


def split_dataframe(fnames, pname, isEFT=False):
    fname = fnames[0]
    with gzip.open(fname, 'rb') as f:
        df = cloudpickle.load(f)
    # Randomize dataset
    df = df.sample(frac=1)
    # Split 95:5 ratio (training:validation)
    train_index = round(len(df) * 0.95)
    train = df[:train_index]
    validation = df[train_index:]
    # Write out validation
    if isEFT: dname = 'eft'
    else: dname = 'powheg'
    validation.to_pickle(f'{pname}/validation_{dname}_dataframe.pkl')
    
    return train


with open('Inputs/xsec_semilep.json') as f:
    xsecs = json.load(f)
with open('ttbar_utilities/inputs/luminosity_in_fb.json') as f:
    lumis = json.load(f)
    

def convert_to_net_tensors(config, dataframe, isEFT=False, doWgtNorm=True):
    # Extract features
    features = from_numpy(dataframe[config['features']].values).to(dtype=float64)
    # Handle if is eft or is powheg
    if isEFT: 
        useWeights = 'SMweights'
        #meta_name = 'signalMetadata'
        sum_name = 'sumSMreweights'
        samp = 'eft'
        labels = tensor([1], dtype=float64).repeat(features.size(0))
    else: 
        useWeights = 'genweights'
        #meta_name = 'backgroundMetadata'
        sum_name = 'sumGenWeights'
        samp = 'powheg'
        labels = tensor([0], dtype=float64).repeat(features.size(0))
    # Extract weights
    wgts = from_numpy(dataframe[useWeights].values).to(dtype=float64)
    # Apply cross-section normalization
    #with open(config[meta_name][0]) as f:
        #metadata = json.load(f)
    sow = dataframe[useWeights].sum()
    norm = xsecs['UL2017'][samp] / sow * lumis['UL2017']*1000
    wgts = wgts * norm
    # Apply training weights normalization
    wgtVal = tensor([1.])
    if doWgtNorm:
        wgts, wgtVal = apply_weight_norm(wgts, config['normalization'])
        config[f'{useWeights}_normVal'] = wgtVal.tolist()

    return TensorDataset(features, wgts, labels), wgtVal

def create_train_dataset(config):
    name = config['name']
    os.makedirs(name, mode=0o755, exist_ok=True)
    if config['normalization'] is None: doWgtNorm = False
    elif config['normalization']==1: doWgtNorm = False
    else: doWgtNorm = True
    # Split dataframes into training and validation
    # Convert dataframes to tensor
    with gzip.open(config['signalSample'][0], 'rb') as f:
        sig_df = cloudpickle.load(f)
    with gzip.open(config['backgroundSample'][0], 'rb') as f:
        bkg_df = cloudpickle.load(f)
    # Remove negative weights
    sig_df = sig_df[sig_df['SMweights']>=0]
    bkg_df = bkg_df[bkg_df['genweights']>=0]
    # Remove events with top mass outside [150,192]
    sig_df = sig_df[(sig_df['top1mass']>=150.) & (sig_df['top1mass']<=192.5)]
    sig_df = sig_df[(sig_df['top2mass']>=150.) & (sig_df['top2mass']<=192.5)]
    bkg_df = bkg_df[(bkg_df['top1mass']>=150.) & (bkg_df['top1mass']<=192.5)]
    bkg_df = bkg_df[(bkg_df['top2mass']>=150.) & (bkg_df['top2mass']<=192.5)]
    # Cap number of events in train/test set
    cap = np.min([int(3e6), sig_df.shape[0], bkg_df.shape[0]])
    print('Keeping {:.2E} events per dataset for training'.format(cap))
    # Convert pandas dataframes to pytorch tensors
    sig_dataset, sig_wgtVal = convert_to_net_tensors(config, sig_df[:cap], isEFT=True, doWgtNorm=doWgtNorm)
    bkg_dataset, bkg_wgtVal = convert_to_net_tensors(config, bkg_df[:cap], isEFT=False, doWgtNorm=doWgtNorm)
    config['sig_wgtVal'] = sig_wgtVal.tolist()
    config['bkg_wgtVal'] = bkg_wgtVal.tolist()
    # Combine datasets into single
    if config['normalization']==1:
        dataset = TensorDataset(
            cat([bkg_dataset[:][0][:cap], sig_dataset[:][0][:cap]]),
            tensor([1.], dtype=float64).repeat(2*cap),
            cat([bkg_dataset[:][2][:cap], sig_dataset[:][2][:cap]]),
    )
    else:
        dataset = TensorDataset(
            cat([bkg_dataset[:][0][:cap], sig_dataset[:][0][:cap]]),
            cat([bkg_dataset[:][1][:cap], sig_dataset[:][1][:cap]]),
            cat([bkg_dataset[:][2][:cap], sig_dataset[:][2][:cap]]),
        )
    # Normalize features
    features, fmeans, fstds = apply_feature_norm(dataset[:][0])
    config['fmeans'] = fmeans.tolist()[0]
    config['fstds']  = fstds.tolist()[0]
    # Split dataset into training and testing
    train, test = random_split(TensorDataset(features, dataset[:][1], dataset[:][2]), [0.8, 0.2])
    print('Training sample has {:.2E} events'.format(train[:][1].shape[0]))
    print('Testing sample has {:.2E} events'.format(test[:][1].shape[0]))
    config['traindata'] = f'{name}/train_tensor.pkl'
    save(train, config['traindata'])
    config['testdata'] = f'{name}/test_tensor.pkl'
    save(test, config['testdata'])

    del train, test, dataset
    # Update config
    with open(config['name']+'/training.yml', 'w') as f:
        yaml.dump(config, f)
    return config