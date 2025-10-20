from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import clf, close, figure, subplots, subplots_adjust
import matplotlib.table as tbl
from os import makedirs
from torch import save
from yaml import dump, safe_load
import numpy as np
from metrics import netEval

def histPlot(background, signal, label, outname=None, backgroundWeights=None, signalWeights=False, ylog=False, xlog=False):
    fig, ax = subplots(figsize=[12,8])
    if xlog:
        bins = np.logspace(np.log10(np.min([background[background > 0] .min(), signal[signal > 0].min()])), 
                           np.log10(np.max([background.max(), signal.max()])),
                           200
                          )
    else:
        bins = np.linspace(np.min([background.min(), signal.min()]), 
                           np.max([background.max(), signal.max()]),
                           200
                          )
    if backgroundWeights:
        ax.hist(background, weights=backgroundWeights, bins=bins, label='background', histtype='step')
    else:
        ax.hist(background, bins=bins, label='background', histtype='step')
    if signalWeights:
        ax.hist(signal, weights=signalWeights, bins=bins, label='signal', histtype='step')
    else:
        ax.hist(signal, bins=bins, label='signal', histtype='step')
    ax.set_xlabel(label, fontsize=12)
    ax.legend()

    parametric_table(dedicated, parametric, ax)
    if ylog:
        ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')
    if outname:
        fig.savefig(f'{outname}')
        clf()
        close()
    else:
        fig.show()

def ratioPlot(x, dedicatedLR, parametricLR, eftCoeffs, bins, wcs, outname=None, 
              plotLog=False, ratioLog=False, xlabel=None, showNoWeights=True, density=False):
    from hist.axis import Regular, StrCategory
    from topcoffea.modules.histEFT import HistEFT
    ax  = []
    fig = figure(figsize=(12,9))
    gs  = GridSpec(6,6, figure=fig)
    
    [terms,values] = zip(*wcs.items())
    
    histEFT = HistEFT(StrCategory(['histEFT'], name='category'),
                      Regular(
                          start=min(bins),
                          stop=max(bins),
                          bins=len(bins) - 1,
                          name="kin",
                          label='HistEFT'
                      ),
                      wc_names=terms
                     )

    ax.append(fig.add_subplot(gs[0:5,0:5]))
    ax.append(fig.add_subplot(gs[5,0:5]))
    subplots_adjust(hspace=0.2)

    histEFT.fill(kin=x, eft_coeff=eftCoeffs, category='histEFT')

    histEFTEval = histEFT.as_hist(values)
    histEFTEval.plot1d(ax=ax[0], density=density, yerr=False)
    nDedicated,_,_  = ax[0].hist(x, bins=bins, weights=dedicatedLR, label='Dedicated', histtype='step', density=density)
    nParametric,_,_ = ax[0].hist(x, bins=bins, weights=parametricLR, label='Parametric', histtype='step', density=density, linestyle='dashdot')
    if showNoWeights: 
        ax[0].hist(x, bins=bins, label='No Weights', histtype='step', color='k', linestyle='dashed', density=density)
    ax[0].legend()
    ax[0].set_xlabel('')
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('')
    if plotLog:
        ax[0].set_yscale('log')
    ax[0].autoscale() 

    if density:
        nHistEft = (histEFTEval.values().flatten()/(sum(histEFTEval.values().flatten())*np.diff(bins)))
        dedicatedRatio = np.ones(nHistEft.shape)
        dedicatedRatio[nHistEft != 0] = nDedicated[nHistEft != 0]/nHistEft[nHistEft != 0]
        parametricRatio = np.ones(nHistEft.shape)
        parametricRatio[nHistEft != 0] = nParametric[nHistEft != 0]/nHistEft[nHistEft != 0]
    else: 
        nHistEft = histEFTEval.values().flatten()
        mask = (nHistEft != 0) & (nParametric > 0)
        dedicatedRatio = np.ones(nHistEft.shape)
        dedicatedRatio[mask] = nDedicated[mask]/nHistEft[mask]
        parametricRatio = np.ones(nHistEft.shape)
        parametricRatio[mask] = nParametric[mask]/nHistEft[mask]
        
    ax[1].hlines(1,ax[0].get_xlim()[0], ax[0].get_xlim()[1], color='k', linestyle='dashed')
    ax[1].plot((bins[1:] + bins[:-1])/2, dedicatedRatio, '^', label='Dedicated', color='orange')
    ax[1].plot((bins[1:] + bins[:-1])/2, parametricRatio, 'v', label='Parametric', color='green')
    ax[1].legend() 

    if ax[1].get_ylim()[0] > 0:
        order = max([np.log10(ax[1].get_ylim()[1]), abs(np.log10(ax[1].get_ylim()[0]))])
    else: 
        lEdge = np.min((dedicatedRatio[dedicatedRatio != 0].min(), parametricRatio[parametricRatio != 0].min()))
        uEdge = np.max((dedicatedRatio[dedicatedRatio != 0].max(), parametricRatio[parametricRatio != 0].max()))
        order = np.max((abs(np.log10(lEdge)), abs(np.log10(uEdge))))
    if ratioLog or (abs(order) > 1):
        ax[1].set_yscale('log')
        ax[1].set_ylim(10**(-order), 10**order)
    else:
        deviation = max([ax[1].get_ylim()[1] - 1, 1- ax[1].get_ylim()[0]])
        if deviation < 1:
            ax[1].set_ylim(1-deviation, 1+deviation)
        else:
            ax[1].set_ylim(0, 1+deviation)
    if xlabel:
        ax[1].set_xlabel(xlabel, fontsize=12)
    ax[1].set_xlim(ax[0].get_xlim())
    return fig, ax
    if outname:
        fig.savefig(f'{outname}')
        clf()
        close()
    else:
        fig.show()

def parametric_table(dedicated, config):
    # Opening the config files
    with open(dedicated+"/training.yml", 'r') as f:
        ded = safe_load(f)
    parametric = []
    for net in config['networks']:
        with open(net+"/training.yml", 'r') as f:
            parametric.append(safe_load(f))
    # Checking that the configs match
    keys = ['backgroundTrainingPoint', 'startingPoint', 'wcs']
    for key in keys:
        for net in parametric:
            if net[key] != ded[key]:
                print(f'mismatch of {key} for {net["name"]}')
                break
    # Preparing the data
    rows = ['cSM']+ded['wcs']
    columns = ['gen', 'ref', 'ded']
    for i in range(len(parametric)):
        columns.append(f'par{i}')
    key = 'signalTrainingPoint'
    data = [np.array(ded['startingPoint']).T,
            np.array(ded['backgroundTrainingPoint']).T,
            np.array(ded['signalTrainingPoint']).T]
    for net in parametric:
        data.append(np.array(net[key]).T)
    data = np.stack(np.array(data),axis=1)

    colors = np.array([np.array([0]*len(rows))]*len(columns))
    colors = np.where(abs(data)>0, 'cyan', 'w')
    return data, rows, columns, colors
    # Making the table
    the_table = tbl.table(cellText=data, rowLabels=rows, colLabels=columns,
                          cellLoc='center', loc='right',
                          colWidths=np.ones(len(columns))*0.1
                         )
    return the_table