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
    # Calculate weights for train sample at various points
    trainBW = train[:][1]@expandArray(config['backgroundPoint']).to(device=config['device'])
    trainSW = train[:][1]@expandArray(config['signalPoint']).to(device=config['device'])
    trainRW = train[:][1]@expandArray(config['referencePoint']).to(device=config['device'])
    # Calculate weights for test sample at various points
    testBW  = test[:][1]@expandArray(config['backgroundPoint']).to(device=config['device'])
    testSW  = test[:][1]@expandArray(config['signalPoint']).to(device=config['device'])
    testRW  = test[:][1]@expandArray(config['referencePoint']).to(device=config['device'])

    # Only positive weights
    train_noNeg = (trainBW>=0) & (trainSW>=0) & (trainRW>=0)
    test_noNeg  = (testBW>=0) & (testSW>=0) & (testRW>=0)
    trainBW = trainBW[train_noNeg]
    trainSW = trainSW[train_noNeg]
    trainRW = trainRW[train_noNeg]
    testBW  = testBW[test_noNeg]
    testSW  = testSW[test_noNeg]
    testRW  = testRW[test_noNeg]

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
    train = TensorDataset(train[:][0][train_noNeg], trainBW/(trainRW*config['bkg2ref']), trainSW/(trainRW*config['sig2ref']))
    test  = TensorDataset(test[:][0][test_noNeg],  testBW/(testRW*config['bkg2ref']),   testSW/(testRW*config['sig2ref']))
    return train, test