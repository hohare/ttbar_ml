import numpy as np
import matplotlib.pyplot as plt
from torch import linspace
import hist
import mplhep
import yaml
from os import makedirs

from training.metrics import netEval
import utils.plotting_conventions as myplt

def plot_losses(testLoss_lst, trainLoss_lst, label, epoch):
    makedirs(f'{label}', mode=0o755, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=[8,5])
    ax.plot( range(len(testLoss_lst)), trainLoss_lst, label="Training dataset")
    ax.plot( range(len(testLoss_lst)), testLoss_lst , label="Testing dataset")
    ax.legend()
    ax.set_xlabel('Loss')
    fig.savefig(f'{label}/training/epoch{epoch}_loss.png')
    plt.close()

def plot_network(network, testset, name, epoch):
    label = f'{name}/training/'
    makedirs(f'{label}', mode=0o755, exist_ok=True)

    test_net = network(testset[:][0]).ravel().detach().cpu().numpy()
    test_bkgnd_weights = testset[:][1].detach().cpu().numpy(),
    test_signal_weights = testset[:][2].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=[8,5])
    bins = linspace(0,1,100)
    # plot test as filled hist
    ax.hist(test_net, weights=test_bkgnd_weights, 
            bins=bins, alpha=0.5, label='Background')
    ax.hist(test_net, weights=test_signal_weights,
            bins=bins, alpha=0.5, label='Signal');

    ax.set_xlabel('network output')
    ax.legend()
    fig.savefig(f'{label}/epoch{epoch}_netOut.png')
    ax.set_yscale('log')
    fig.savefig(f'{label}/epoch{epoch}_netOut_log.png')
    plt.clf()
    plt.close()

def get_ratio(numer, denom):
    mask = (numer > 0) & (denom > 0)
    ratio = np.ones(numer.shape)
    ratio[mask] = numer[mask]/denom[mask]
    return ratio

def plot_network_end(network, testset, trainset, testLoss, trainLoss, name, epoch):
    label = f'{name}/complete/'
    makedirs(f'{label}', mode=0o755, exist_ok=True)

    #LOSS
    fig, ax = plt.subplots(1, 1, figsize=[8,5])
    ax.plot( range(len(testLoss)), trainLoss, label="Training dataset")
    ax.plot( range(len(testLoss)), testLoss, label="Testing dataset")
    ax.legend()
    ax.set_xlabel('Loss')
    fig.savefig(f'{label}/loss_epoch{epoch}.png')
    plt.close()
    
    #ROC CURVE
    test_output  = network(testset[:][0])
    backgroundWeights = testset[:][1]#.detach()
    signalWeights = testset[:][2]#.detach()
    fpr, tpr, auc, a = netEval(test_output, backgroundWeights, signalWeights)

    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    ax.plot(fpr, tpr, label='Network Performance')
    ax.plot([0,1],[0,1], ':', label='Baseline')
    ax.legend()
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'AUC: {auc}', fontsize=14)
    fig.savefig(f'{label}/roc.png')
    plt.close()

    #SAVE PERFORMANCE METRICS
    performance = {
        'Area under ROC': auc,
        'Accuracy': a
    }
    with open(f'{label}/performance.yml','w') as f:
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
    histo.fill(
        net = test_output.ravel().detach().cpu().numpy(),
        dataset = "test",
        event = 0,
        weight = testset[:][1].detach().cpu().numpy()
    )
    histo.fill(
        net = test_output.ravel().detach().cpu().numpy(),
        dataset = "test",
        event = 1,
        weight = testset[:][2].detach().cpu().numpy()
    )
    test_net_bkgnd = histo[{"dataset":"test","event":0}].values()
    mplhep.histplot(test_net_bkgnd/test_net_bkgnd.sum(), 
                    bins=bins, alpha=0.5, label="Background (test)", histtype="fill", 
                    yerr=0, color=myplt.colors[0])
    test_net_signal = histo[{"dataset":"test","event":1}].values()
    mplhep.histplot(test_net_signal/test_net_signal.sum(), 
                    bins=bins, alpha=0.5, label="Signal (test)", histtype="fill", 
                    yerr=0, color=myplt.colors[1])
                    
    # plot train as dots
    train_output  = network(trainset[:][0])
    histo.fill(
        net = train_output.ravel().detach().cpu().numpy(),
        dataset = "test",
        event = 0,
        weight = trainset[:][1].detach().cpu().numpy()
    )
    histo.fill(
        net = train_output.ravel().detach().cpu().numpy(),
        dataset = "test",
        event = 1,
        weight = trainset[:][2].detach().cpu().numpy()
    )

    train_net_bkgnd = histo[{"dataset":"test","event":0}].values()
    mplhep.histplot(train_net_bkgnd/train_net_bkgnd.sum(), 
                    bins=bins, alpha=0.5, label="Background (train)", histtype="errorbar",
                    yerr=0)
    train_net_signal = histo[{"dataset":"test","event":1}].values()
    mplhep.histplot(train_net_signal/train_net_signal.sum(), 
                    bins=bins, alpha=0.5, label="Signal (train)", histtype="errorbar",
                    yerr=0)

    axes.append(fig.add_subplot(gs[1]))
    plt.subplots_adjust(hspace=0.2)    
    bins = histo.axes[0].centers
    plt.grid('both')
    axes[1].hlines(1, bins[0], bins[-1], color='grey', ls='--', lw=1)
    bratio = get_ratio(test_net_bkgnd/test_net_bkgnd.sum(), train_net_bkgnd/train_net_bkgnd.sum())
    sratio = get_ratio(test_net_signal/test_net_signal.sum(), train_net_signal/train_net_signal.sum())
    axes[1].scatter(bins, bratio, marker='^', color='blue',)
    axes[1].scatter(bins, sratio, marker='v', color='orange',)
    
    axes[0].set_xlabel('')
    axes[0].set_title('network output', fontsize=12)
    axes[0].set_ylabel('Normalized', fontsize=18)
    axes[1].set_ylabel('test / train', fontsize=18)

    axes[0].legend(ncols=2)
    
    fig.savefig(f'{label}/netOut_epoch{epoch}.png')
    axes[0].set_yscale('log')
    fig.savefig(f'{label}/netOut_epoch{epoch}_log.png')
    plt.close()
    return fig, axes