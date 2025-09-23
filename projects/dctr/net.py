import torch.nn as nn
from torch import float64
from yaml import safe_load

cost =  nn.BCELoss(reduction='mean')

def createModel(nFeatures, network):
    with open(f'projects/dctr/networks/{network}.yml', 'r') as f:
        config = safe_load(f)
    layers = []
    for i, layer in enumerate(config['model']['layers']):
        layerType = layer['type']
        if i == 0:
            if layerType == 'Linear':
                layers.append(nn.Linear(nFeatures, layer['out']))
        else:
            if layerType == 'Linear':
                layers.append(nn.Linear(layer['in'], layer['out']))
        if layer['activation'] == 'LeakyReLU':
            layers.append(nn.LeakyReLU())
        elif layer['activation'] == 'Sigmoid':
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self, nFeatures, device, network):
        super().__init__()
        if network:
            self.main_module = createModel(nFeatures, network)
        else:
            self.main_module = nn.Sequential( 
                nn.Linear(nFeatures, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        self.main_module.type(float64)
        self.main_module.to(device)
    def forward(self, x):
        return self.main_module(x)
    
class Model:
    def __init__(self, nFeatures, device, network):
        '''
        features: inputs used to train the neural network
        device: device used to train the neural network
        network: network structure used to train the neural network
        '''
        self.net = Net(nFeatures, device=device, network=network)
        cost.to(device)
        
    def loss(self, features, weights, targets):
        cost.weight = weights
        return cost(self.net(features).squeeze(), targets)