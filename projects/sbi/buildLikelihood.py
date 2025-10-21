from net import Net
from weight_manager import expandArray
from torch import device, load, tensor, vstack, float64, no_grad
import numpy as np
from torch.linalg import lstsq
from yaml import safe_load

class likelihood:
    def __init__(self, config, nFeatures, network=None):
        with open(config+"/training.yml") as f:
            self.config = safe_load(f)
        self.net = Net(nFeatures, self.config['device'], self.config['network'])
        self.net.load_state_dict(load(f'{self.config["name"]}/networkStateDict.p', 
                                        map_location=device(self.config['device'])))
    def __call__(self, features, network=None):
        #means = tensor(self.config['trainmeans']).to(device=device(self.config['device']))
        #stds = tensor(self.config['trainstds']).to(device=device(self.config['device']))
        with no_grad():
            s   = self.net((features - features.mean(0))/features.std(0))
        lr  = (s/(1-s)).flatten()
        lr *= self.config['sig2bkg']
        return lr
            
class fullLikelihood: 
    def __init__(self, config, features):
        self.config = config
        trainingMatrix = []
        ratios = []
        # Construct matrices for later fitting
        for i, network in enumerate(self.config['networks']):
            tmp = network.split("\\")[-1]
            print(f'- {tmp}')
            network = likelihood(network, len(self.config['features']))
            trainingMatrix += [expandArray(network.config['signalPoint']).to(device=device(self.config['device']))]
            ratios += [network(features)]
        self.wcs = network.config['wcs']
        trainingMatrix = vstack(trainingMatrix)
        self.zerosMask = ~(trainingMatrix==0).all(dim=0)
        trainingMatrix = trainingMatrix[:,self.zerosMask]
        ratios = vstack(ratios)
        features = None
        self.alphas, self.residuals, self.rank, _ = lstsq(trainingMatrix, ratios, rcond=-1)
    def __call__(self, coefs):
        return expandArray(coefs).to(device=self.config['device'])[self.zerosMask]@self.alphas