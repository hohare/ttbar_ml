from torch import tensor, load, device, float64, transpose, from_numpy
from training.normalizations import apply_feature_norm
import numpy as np
from projects.dctr.preprocess import prep_dataset
from projects.dctr.reweighting import load_network, compute
from torch.utils.data import TensorDataset
    
def calculate_reweight(top_info, config, isEFT):
    # Load network
    model = load_network(config)
    # Convert to tensor form
    arr = np.array(list(top_info.values()))
    features = transpose(from_numpy(arr).to(dtype=float64), 0,1)
    # Normalize features
    features = prep_dataset(TensorDataset(features), config, validation=True)
    # Get network prediction and predict factor
    factor = compute(model, features)
    if isEFT:
        factor = 1/factor

    return factor.flatten().detach().numpy()