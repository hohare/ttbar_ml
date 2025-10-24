from argparse import ArgumentParser
import numpy as np
from os import makedirs, system
from torch import device, load
from yaml import dump, safe_load
import matplotlib.pyplot as plt
import matplotlib.table as tbl
from matplotlib.pyplot import clf, close, figure, subplots, subplots_adjust
from matplotlib.gridspec import GridSpec

from buildLikelihood import fullLikelihood, likelihood
from validation_plotting import parametric_table

def plot_r(x, dedicatedLR, eftCoeffs, bins, wcs,):
    import hist
    h = hist.Hist(
        #hist.axes.Regular(0, 20., ''),
    )
    

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
    if parametricLR:
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
        if parametricLR:
            parametricRatio = np.ones(nHistEft.shape)
            parametricRatio[nHistEft != 0] = nParametric[nHistEft != 0]/nHistEft[nHistEft != 0]
    else: 
        nHistEft = histEFTEval.values().flatten()
        if parametricLR:
            mask = (nHistEft != 0) & (nParametric > 0)
            parametricRatio = np.ones(nHistEft.shape)
            parametricRatio[mask] = nParametric[mask]/nHistEft[mask]
        else:
            mask = (nHistEft != 0) 
        dedicatedRatio = np.ones(nHistEft.shape)
        dedicatedRatio[mask] = nDedicated[mask]/nHistEft[mask]
        
    ax[1].hlines(1,ax[0].get_xlim()[0], ax[0].get_xlim()[1], color='k', linestyle='dashed')
    ax[1].plot((bins[1:] + bins[:-1])/2, dedicatedRatio, '^', label='Dedicated', color='orange')
    if parametricLR:
        ax[1].plot((bins[1:] + bins[:-1])/2, parametricRatio, 'v', label='Parametric', color='green')
    ax[1].legend() 

    if ax[1].get_ylim()[0] > 0:
        order = max([np.log10(ax[1].get_ylim()[1]), abs(np.log10(ax[1].get_ylim()[0]))])
    elif parametricLR: 
        lEdge = np.min((dedicatedRatio[dedicatedRatio != 0].min(), parametricRatio[parametricRatio != 0].min()))
        uEdge = np.max((dedicatedRatio[dedicatedRatio != 0].max(), parametricRatio[parametricRatio != 0].max()))
        order = np.max((abs(np.log10(lEdge)), abs(np.log10(uEdge))))
    else:
        order = abs(np.log10(dedicatedRatio[dedicatedRatio != 0].max()))
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

def main(dedicated):
    with open(dedicated+"training.yml", 'r') as f:
        config = safe_load(f)
    with open("projects/sbi/validation/features.yml",'r') as f:
        feature_list = safe_load(f)
    output = config['name']

    doMask=True
    
    makedirs(output, mode=0o755, exist_ok=True)
    
    print('Loading data...')
    dataset = load(f"{config['data']}/validation.p", map_location=device(config['device']), weights_only=False)

    print('Preparing dedicated likelihood...')
    dlr_obj  = likelihood(dedicated, dataset[:][0].shape[1])
    dWcs = dlr_obj.config['signalPoint']
    wcDict = {}
    for i, wc in enumerate(dlr_obj.config['wcs']):
        wcDict[wc] = dWcs[i+1]
    dlr = dlr_obj(dataset[:][0], dWcs).detach().cpu().numpy()
    #s, dlr = dlr_obj(dataset[:][0], dWcs)
    #s=s.detach().cpu().numpy()
    #dlr=dlr.detach().cpu().numpy()
    #print(np.min(s), np.max(s))
    
    features = dataset[:][0].detach().cpu().numpy()
    fitCoefs = dataset[:][1].detach().cpu().numpy()

    folder_name = 'dedicated'
    if doMask:
        maxS = 0.99
        mask = (s<=maxS)
        dlr = dlr[mask]
        print('new max',np.max(s[mask]))
        features = features[mask]
        fitCoefs = fitCoefs[mask]
        folder_name += f'_masked_{str(maxS)}'

    output = f'{output}/{folder_name}'
    makedirs(f'{output}/noNorm',  mode=0o755, exist_ok=True)
    makedirs(f'{output}/density', mode=0o755, exist_ok=True)

    fig, ax = plt.subplots()
    plt.hist(dlr, bins=100, label='dlr')
    ax.set_title(folder_name)
    ax.set_xlabel('dlr')
    ax.set_yscale('log')
    fig.savefig(f'{output}/dlr_log.png', bbox_inches='tight')
    plt.show()
    plt.close()

    data, rows, cols, colors = parametric_table(dedicated, None)
    for kinematic, params in feature_list["features"].items():
        if "jet2" in kinematic: continue
        elif "jet3" in kinematic: continue
        elif "jet4" in kinematic: continue
        print(f'- making plots for kinematic {kinematic}')
        bins = np.linspace(params['min'], params['max'], params['nbins'])
        
        fig, ax = ratioPlot(features[:,params['loc']], dlr, None, fitCoefs, bins, wcDict,
                  xlabel=params['label'])
        tbl.table(ax[0], cellText=data, rowLabels=rows, colLabels=cols,
                 cellLoc='center', loc='top', cellColours=colors, colWidths=np.ones(len(cols))*0.1,)
        fig.savefig(f'{output}/noNorm/{kinematic}.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        fig, ax = ratioPlot(features[:,params['loc']], dlr, None, fitCoefs, bins, wcDict,
                  xlabel=params['label'], plotLog=True)
        tbl.table(ax[0], cellText=data, rowLabels=rows, colLabels=cols,
                 cellLoc='center', loc='top', cellColours=colors, colWidths=np.ones(len(cols))*0.1,)
        fig.savefig(f'{output}/noNorm/{kinematic}_log.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        fig, ax = ratioPlot(features[:,params['loc']], dlr, None, fitCoefs, bins, wcDict,
                  xlabel=params['label'], density=True)
        tbl.table(ax[0], cellText=data, rowLabels=rows, colLabels=cols,
                 cellLoc='center', loc='top', cellColours=colors, colWidths=np.ones(len(cols))*0.1,)
        fig.savefig(f'{output}/density/{kinematic}.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        fig, ax = ratioPlot(features[:,params['loc']], dlr, None, fitCoefs, bins, wcDict,
                  xlabel=params['label'], density=True, plotLog=True)
        tbl.table(ax[0], cellText=data, rowLabels=rows, colLabels=cols,
                 cellLoc='center', loc='top', cellColours=colors, colWidths=np.ones(len(cols))*0.1,)
        fig.savefig(f'{output}/density/{kinematic}_log.png', bbox_inches='tight')
        plt.show()
        plt.close()

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('dedicated', help = 'path to folder used for training')
    args = parser.parse_args()
    main(args.dedicated)