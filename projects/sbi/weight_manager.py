import yaml
from numpy import array, float32
from torch.utils.data import TensorDataset
from torch import tensor, float64

def expandArray(coefs):
    arrayOut = []
    for i in range(len(coefs)):
         for j in range(i+1):
             arrayOut += [coefs[i]*coefs[j]]
    return array(arrayOut).astype(float32)

def calculate_weights(train, test, config):
    print('Calculating weights...')
    # Calculate weights for train sample at various points
    trainBW = train[:][1]@tensor(expandArray(config['backgroundTrainingPoint']), dtype=float64).to(device=config['device'])
    trainSW = train[:][1]@tensor(expandArray(config['signalTrainingPoint']),     dtype=float64).to(device=config['device'])
    trainGW = train[:][1]@tensor(expandArray(config['startingPoint']),           dtype=float64).to(device=config['device'])
    trainSM = train[:][1]@tensor(expandArray([1] + [0]*(len(config['signalTrainingPoint']) - 1)), dtype=float64).to(device=config['device'])
    # Calculate weights for test sample at various points
    testBW  = test[:][1]@tensor(expandArray(config['backgroundTrainingPoint']), dtype=float64).to(device=config['device'])
    testSW  = test[:][1]@tensor(expandArray(config['signalTrainingPoint']),     dtype=float64).to(device=config['device'])
    testGW  = test[:][1]@tensor(expandArray(config['startingPoint']),           dtype=float64).to(device=config['device'])
    testSM  = test[:][1]@tensor(expandArray([1] + [0]*(len(config['signalTrainingPoint']) - 1)), dtype=float64).to(device=config['device'])

    # Useful average weight ratios across train and test
    nEvents = testSW.shape[0] + trainSW.shape[0]    
    config['sig2gen'] = (((trainSW/trainGW).sum() + (testSW/testGW).sum())/nEvents).item()
    config['bkg2gen'] = (((trainBW/trainGW).sum() + (testBW/testGW).sum())/nEvents).item()
    config['sig2bkg'] = (((trainSW/trainBW).sum() + (testSW/testBW).sum())/nEvents).item()
    config['bkg2sm']  = (((trainBW/trainSM).sum() + (testBW/testSM).sum())/nEvents).item()
    # Update config with the results
    with open(config['name']+'/training.yml', 'w') as f:
        yaml.dump(config, f)

    # Construct the datasets to be (features, backgroundWeights_normalized, signalWeights_normalized)
    train = TensorDataset(train[:][0], trainBW/(trainGW*config['bkg2gen']), trainSW/(trainGW*config['sig2gen']))
    test  = TensorDataset(test[:][0],  testBW/(testGW*config['bkg2gen']),   testSW/(testGW*config['sig2gen']))
    return train, test