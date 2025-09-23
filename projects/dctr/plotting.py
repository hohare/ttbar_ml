import os
import hist
import mplhep
import matplotlib.pyplot as plt
import numpy as np

default_cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_ratio(numer, denom):
    mask = (numer > 0) & (denom > 0)
    ratio = np.ones(numer.shape)
    ratio[mask] = numer[mask]/denom[mask]
    return ratio

def get_chi2(exp, obs):
    mask = (exp > 0) & (obs > 0)
    chi2 = np.ones(exp.shape)
    chi2[mask] = ((np.power(obs-exp, 2)/exp)/len(obs))[mask]
    return chi2

def get_delta_ratio(initial, reweight):
    mask = (initial == 1.) & (reweight == 1.)
    dr = (np.abs(initial-1)-np.abs(reweight-1))/np.abs(initial-1)
    dr[mask] = 0.
    return dr

def plot_reweight(histo, config, makeLog=False, 
                  name="", runSave=False, path=None,
                  fig=None, axes=None, density=False, **kwargs
                 ):
    """Plot background, signal, and reweighting"""
    if fig is None: makeit = True
    else: makeit=False
    if runSave:
        if path is None: path=config['name']+"/plots"
        os.makedirs(path, mode=0o755, exist_ok=True)

    if makeit:
        axes = []
        if "altRatio" in kwargs: 
            figsize=(8,7)
            nplots = 3
            height_ratios=[2,1,1]
        else: 
            figsize=(6,5)
            nplots = 2
            height_ratios=[3,1]
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(nplots,1, height_ratios=height_ratios)
        axes.append(fig.add_subplot(gs[0]))
        axes.append(fig.add_subplot(gs[1]))
        plt.subplots_adjust(hspace=0.2)
    
    bkgnd = histo[{"dataset":0, "postrwgt":0}]
    mplhep.histplot(bkgnd, label='Powheg', density=density, histtype='fill', alpha=0.4, yerr=0, ax=axes[0]);
    signal = histo[{"dataset":1, "postrwgt":0}]
    mplhep.histplot(signal, label='SMEFTsim', density=density, histtype='fill', alpha=0.4, yerr=0, ax=axes[0]);
    sig2bkgnd = histo[{"dataset":1, "postrwgt":1}]
    mplhep.histplot(sig2bkgnd, label='SMEFTsim rwgt to Powheg', density=density, color='blue', yerr=0, ls='--', ax=axes[0])
    bkgnd2sig = histo[{"dataset":0, "postrwgt":1}]
    mplhep.histplot(bkgnd2sig, label='Powheg rwgt to SMEFTsim', density=density, color='orange', yerr=0, ls='--', ax=axes[0])

    bins = bkgnd.axes[0].centers
    mask = (signal.values() > 0) & (bkgnd.values() > 0)
    bratio  = get_ratio(sig2bkgnd.values(), bkgnd.values())
    bsratio = get_ratio(   signal.values(), bkgnd.values())
    sratio  = get_ratio(bkgnd2sig.values(), signal.values())
    sbratio = get_ratio(    bkgnd.values(), signal.values())

    #default ratio
    axes[1].scatter(bins, bratio, marker='^', color='blue');
    axes[1].scatter(bins, sratio, marker='v', color='orange');
    axes[1].scatter(bins, bsratio, s=8, marker='+', color=default_cycler[0]);
    axes[1].scatter(bins, sbratio, s=4, marker='x', color='#ffc35b');
    axes[1].set_ylabel(r'$r =$ rwgt / target', fontsize=14)

    #axes[1].set_ylim(np.min([bratio, sratio]), np.max([bratio,sratio]))
    plt.grid('both')

    feature = histo.axes[0].label
    axes[0].set_xlabel('')

    if "altRatio" in kwargs:
        axes.append(fig.add_subplot(gs[2]))
        plt.subplots_adjust(hspace=0.2)
        dr_bkg = get_delta_ratio(bsratio, bratio)
        dr_sig = get_delta_ratio(sbratio, sratio)
        axes[2].scatter(bins, dr_bkg,  color='blue')
        axes[2].scatter(bins, dr_sig,  color='orange')
        
        axes[2].set_ylabel(r'$\frac{\Delta |r-1|}{|r-1|}$', fontsize=16)
        plt.grid('both')  
        axes[2].set_xlabel(feature, fontsize=16)
    else:
        axes[1].set_xlabel(feature, fontsize=16)

    if 'xlim' in kwargs:
        axes[0].set_xlim(kwargs['xlim'])
        if "altRatio" in kwargs:
            axes[2].set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:   axes[0].set_ylim(kwargs['ylim'])
    if 'ylimr' in kwargs:  axes[1].set_ylim(kwargs['ylimr'])
    if 'ylimr2' in kwargs: axes[2].set_ylim(kwargs['ylimr2'])
    
    axes[0].legend(ncols=2);
    #axes[1].legend(['Powheg/SMEFTsim rwgt', 'SMEFTsim/Powheg rwgt'], fontsize=8)
    if runSave: 
        fig.savefig(path+f'/{feature}{name}.png')
        plt.close()
    if makeLog:
        axes[0].set_yscale('log')
        if runSave: 
            fig.savefig(path+f'/{feature}_log{name}.png')
            plt.close()
    return fig, axes