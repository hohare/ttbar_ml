from torch import Tensor

def apply_feature_norm(features, means=None, stds=None):
    if means is None:
        means = features.mean(axis=0, keepdim=True)
    if stds is None:
        stds  = features.std(axis=0, keepdim=True)
    features -= Tensor(means)
    features /= Tensor(stds)
    return features, means, stds

def reverse_feature_norm(features, means, stds):
    features *= Tensor(stds)
    features += Tensor(means)
    return features

def apply_weight_norm(weights, config):
    if config == 'weightMean':
        val = weights.mean()
        weights /= val
    elif config == 'unity':
        val = weights.max()
        weights /= val
    return weights, val

def reverse_weight_norm(weights, targets, config):
    weights = torch.where(targets, weights*config['wBkgNorm'], weights*config['wSigNorm'])
    return weights