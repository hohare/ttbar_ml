from training.net import Net
from training.weight_manager import expandArray
from torch import device, load, tensor, vstack, float64
import numpy as np
from torch.linalg import lstsq
from yaml import safe_load

class likelihood:
    def __init__(self, config, nFeatures, network=None):
        with open(config+"/training.yml") as f:
            self.config = safe_load(f)
        self.net = Net(nFeatures, self.config['device'], self.config['network'])
        self.net.load_state_dict(load(f'{self.config["name"]}/complete/networkStateDict.p', 
                                        map_location=device(self.config['device'])))
    def __call__(self, features, network=None):
        s   = self.net((features - features.mean(0))/features.std(0))
        lr  = (s/(1-s)).flatten()
        lr *= self.config['sig2bkg']
        return lr
            
class fullLikelihood: 
    def __init__(self, config, features):
        self.config = config
        trainingMatrix = []
        ratios = []
        for i, network in enumerate(self.config['networks']):
            tmp = network.split("\\")[-1]
            print(f'- {tmp}')
            network = likelihood(network, len(self.config['features']))
            trainingMatrix += [tensor(expandArray(network.config['signalTrainingPoint']),device=self.config['device'], dtype=float64)]
            ratios += [network(features)]
        self.wcs = network.config['wcs']
        trainingMatrix = vstack(trainingMatrix)
        ratios = vstack(ratios)
        features = None
        self.alphas, _, _, _ = lstsq(trainingMatrix, ratios, rcond=-1)
    def __call__(self, coefs):
        return tensor(expandArray(coefs),device=self.config['device'], dtype=float64)@self.alphas