import numpy as np
import matplotlib.pyplot as plt
from torch import linspace
import hist
import mplhep
import yaml
from torch.utils.data import DataLoader

from .diagnostics import netEval
import training.plotting_conventions as mypltset

def plot_losses(testLoss_lst, trainLoss_lst, name, epoch):
    
    fig, ax = plt.subplots(1, 1, figsize=[8,5])
    ax.plot( range(len(testLoss_lst)), trainLoss_lst, label="Training dataset")
    ax.plot( range(len(testLoss_lst)), testLoss_lst , label="Testing dataset")
    ax.legend()
    ax.set_xlabel('Loss')
    fig.savefig(f'{name}/training/loss_{epoch}.png')
    plt.close()

def plot_network(network, testset, name, epoch):
    backgroundMask = testset[:][2] == 0
    signalMask     = testset[:][2] == 1

    test_bkgnd_net = network(testset[:][0][backgroundMask]).ravel().detach().cpu().numpy()
    test_bkgnd_weights = testset[:][1][backgroundMask].detach().cpu().numpy(),
    test_signal_net = network(testset[:][0][signalMask]).ravel().detach().cpu().numpy()
    test_signal_weights = testset[:][1][signalMask].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=[8,5])
    bins = linspace(0,1,200)
    # plot test as filled hist
    ax.hist(test_bkgnd_net, weights=test_bkgnd_weights, 
            bins=bins, alpha=0.5, label='Background')
    ax.hist(test_signal_net, weights=test_signal_weights,
            bins=bins, alpha=0.5, label='Signal');

    ax.set_xlabel('network output')
    ax.legend()
    fig.savefig(f'{name}/training/netOut_{epoch}.png')
    plt.close()

def get_ratio(numer, denom):
    mask = (numer > 0) & (denom > 0)
    ratio = np.ones(numer.shape)
    ratio[mask] = numer[mask]/denom[mask]
    return ratio

def plot_network_end(network, testset, trainset, name, epoch):
    #ROC CURVE
    test_output  = network(testset[:][0])
    test_weights = testset[:][1].detach()
    signalMask   = (testset[:][2] == 1)
    fpr, tpr, auc, a = netEval(test_output, test_weights, signalMask)

    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    ax.plot(fpr, tpr, label='Network Performance')
    ax.plot([0,1],[0,1], ':', label='Baseline')
    ax.legend()
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'AUC: {auc}', fontsize=14)
    fig.savefig(f'{name}/training/roc.png')
    plt.clf()
    plt.close()

    #SAVE PERFORMANCE METRICS
    performance = {
        'Area under ROC': auc,
        'Accuracy': a
    }
    with open(f'{name}/training/performance.yml','w') as f:
        f.write(yaml.dump(performance))
    
    #PLOTTING THE NETWORK OUTPUTS
    histo = hist.Hist(
        hist.axis.Regular(50, 0., 1., name="net"),
        hist.axis.StrCategory(["test","train"], name="dataset"),
        hist.axis.IntCategory([0,1], name="event"),
        storage="Weight"
    )
    bins = histo.axes['net'].edges

    axes = []
    fig = plt.figure(figsize=(8,8))
    gs = plt.GridSpec(2,1, height_ratios=[3,1])
    axes.append(fig.add_subplot(gs[0]))
    # plot test as filled hist
    signalMask = testset[:][2] == 1
    histo.fill(
        net = test_output.ravel().detach().cpu().numpy(),
        dataset = "test",
        event = signalMask.detach().cpu().numpy(),
        weight = testset[:][1].detach().cpu().numpy()
    )
    test_net_bkgnd = histo[{"dataset":"test","event":0}].values()
    mplhep.histplot(test_net_bkgnd/test_net_bkgnd.sum(), 
                    bins=bins, alpha=0.5, label="Background (test)", histtype="fill", 
                    yerr=0, color=mypltset.colors[0])
    test_net_signal = histo[{"dataset":"test","event":1}].values()
    mplhep.histplot(test_net_signal/test_net_signal.sum(), 
                    bins=bins, alpha=0.5, label="Signal (test)", histtype="fill", 
                    yerr=0, color=mypltset.colors[1])
                    
    # plot train as dots
    train_dl = DataLoader(trainset, batch_size=int(1e5))#this
    for feat, wgt, label in train_dl:
        histo.fill(
            net = network(feat).ravel().detach().cpu().numpy(),
            dataset = "train",
            event = label.detach().cpu().numpy().astype(int),
            weight = wgt.detach().cpu().numpy()
        )

    train_net_bkgnd = histo[{"dataset":"train","event":0}].values()
    mplhep.histplot(train_net_bkgnd/train_net_bkgnd.sum(), 
                    bins=bins, alpha=0.5, label="Background (train)", histtype="errorbar",
                    yerr=0)
    train_net_signal = histo[{"dataset":"train","event":1}].values()
    mplhep.histplot(train_net_signal/train_net_signal.sum(), 
                    bins=bins, alpha=0.5, label="Signal (train)", histtype="errorbar",
                    yerr=0)
    axes[0].legend()
    
    axes.append(fig.add_subplot(gs[1]))
    plt.subplots_adjust(hspace=0.2)    
    bins = histo.axes[0].centers
    plt.grid('both')
    axes[1].hlines(1, bins[0], bins[-1], color='grey', ls='--', lw=1)
    bratio = get_ratio(test_net_bkgnd/test_net_bkgnd.sum(), train_net_bkgnd/train_net_bkgnd.sum())
    sratio = get_ratio(test_net_signal/test_net_signal.sum(), train_net_signal/train_net_signal.sum())
    axes[1].scatter(bins, bratio, marker='^', color='blue', label='Background')
    axes[1].scatter(bins, sratio, marker='v', color='orange', label='Signal')
    
    axes[1].set_xlabel('network output')
    axes[1].set_ylabel('test / train')
    axes[1].legend(ncols=2)
    fig.savefig(f'{name}/training/netOut_{epoch}.png')
    axes[0].set_yscale('log')
    fig.savefig(f'{name}/training/netOut_{epoch}_log.png')
    plt.close()
