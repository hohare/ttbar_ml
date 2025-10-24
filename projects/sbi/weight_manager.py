import yaml
from numpy import array, float32
from torch.utils.data import TensorDataset
from torch import tensor, float64

def expandArray(coefs):
    arrayOut = []
    for i in range(len(coefs)):
         for j in range(i+1):
             arrayOut += [coefs[i]*coefs[j]]
    return tensor(array(arrayOut), dtype=float64)

def calculate_weights(train, test, config):
    print('Calculating weights...')
    factor = 366.358/6063898 * 41.48*1000#xsec/sow * lumi*1000
    # Calculate weights for train sample at various points
    trainBW = train[:][1]@expandArray(config['backgroundPoint']).to(device=config['device'])*factor
    trainSW = train[:][1]@expandArray(config['signalPoint']).to(device=config['device'])*factor
    trainRW = train[:][1]@expandArray(config['referencePoint']).to(device=config['device'])*factor
    # Calculate weights for test sample at various points
    testBW  = test[:][1]@expandArray(config['backgroundPoint']).to(device=config['device'])*factor
    testSW  = test[:][1]@expandArray(config['signalPoint']).to(device=config['device'])*factor
    testRW  = test[:][1]@expandArray(config['referencePoint']).to(device=config['device'])*factor

    # Calculating ratio of means for normalization
    #nEvents = testSW.shape[0] + trainSW.shape[0] #always cancels out
    sigMean = (trainSW.sum() + testSW.sum())#/nEvents
    bkgMean = (trainBW.sum() + testBW.sum())#/nEvents
    refMean = (trainRW.sum() + testRW.sum())#/nEvents
    config['sig2bkg'] = (sigMean/bkgMean).item()
    config['sig2ref'] = (sigMean/refMean).item()
    config['bkg2ref'] = (bkgMean/refMean).item()
    # Update config with the results
    with open(config['name']+'/training.yml', 'w') as f:
        yaml.dump(config, f)

    # Construct the datasets to be (features, backgroundWeights_normalized, signalWeights_normalized)
    #train = TensorDataset(train[:][0], trainBW/(trainRW*config['bkg2ref']), trainSW/(trainRW*config['sig2ref']))
    #test  = TensorDataset(test[:][0],  testBW/(testRW*config['bkg2ref']),   testSW/(testRW*config['sig2ref']))
    train = TensorDataset(train[:][0], trainBW, trainSW)
    test  = TensorDataset( test[:][0],  testBW,  testSW)
    return train, test