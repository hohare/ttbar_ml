from torch import tensor, load, device, float64, transpose
#from torch.utils.data import DataLoader
from training.normalizations import apply_feature_norm
from projects.dctr.net import Net

def load_network(config):
    model = Net(len(config['features']), 'cpu', config['network'])
    name = config['name']
    model.load_state_dict(load(f'{name}/networkStateDict_cpu.p'))#, map_location=device('cpu')))
    model.eval()
    return model
    
def calculate_reweight(top_info, config, isEFT):
    # Load network
    model = load_network(config)
    # Convert to tensor form
    features = transpose(tensor(list(top_info.values())), 0,1).to(dtype=float64)
    # Normalize features
    features, _, _ = apply_feature_norm(features, tensor(config['fmeans']), tensor(config['fstds']))
    # Get network prediction
    s = model(features).flatten().detach()
    # Calcuate ratio 
    factor = s/(1-s)
    if isEFT:
        factor = 1/factor

    #realweights = torch.where(test_targets==1., test_weights/ratios, test_weights*ratios)
    return factor