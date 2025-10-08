import torch

#def netEval(backgroundOutput, signalOutput, backgroundWeights, signalWeights, threshold=0.5, nPoints=200):
def netEval(outputs, backgroundWeights, signalWeights, threshold=0.5, nPoints=200):
    
    bins = torch.linspace(outputs.min().item(),
                          outputs.max().item(),
                          nPoints + 1)
    signalTotal =  signalWeights.sum()
    backgroundTotal = backgroundWeights.sum()
    tpr = []; fpr = []
    for i in range(len(bins)):
        tpr += [    (signalWeights[(outputs >= bins[-(i+1)]).ravel()].sum()/signalTotal).item()]
        fpr += [(backgroundWeights[(outputs >= bins[-(i+1)]).ravel()].sum()/backgroundTotal).item()]

    total = signalTotal + backgroundTotal
    a = ((signalWeights[outputs.ravel() >= threshold].sum() + backgroundWeights[outputs.ravel() <= threshold].sum())/total).item()
    auc = torch.trapz(torch.Tensor(tpr), x=torch.Tensor(fpr)).item()

    return fpr, tpr, auc, a