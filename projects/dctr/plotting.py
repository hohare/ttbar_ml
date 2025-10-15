import os
import hist
import mplhep
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as pltticker

default_cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_ratio(numer, denom):
    mask = (numer > 0) & (denom > 0)
    ratio = np.ones(numer.shape)
    ratio[mask] = numer[mask]/denom[mask]
    return ratio

def get_delta_ratio(initial, reweight):
    mask = (initial == 1.) & (reweight == 1.)
    dr = (np.abs(initial-1)-np.abs(reweight-1))/np.abs(initial-1)
    dr[mask] = 0.
    return dr

def myround(x, base=5):
    return base * round(x/base)

def plot_reweight(histo, config, makeLog=False, plotPowRwgt=True,
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
            figsize=(7,6)
            nplots = 3
            height_ratios=[2,1,1]
        else: 
            figsize=(7,5)
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
    if plotPowRwgt:
        bkgnd2sig = histo[{"dataset":0, "postrwgt":1}]
        mplhep.histplot(bkgnd2sig, label='Powheg rwgt to SMEFTsim', density=density, color='orange', yerr=0, ls='--', ax=axes[0])

    bins = bkgnd.axes[0].centers
    mask = (signal.values() > 0) & (bkgnd.values() > 0)
    bratio  = get_ratio(sig2bkgnd.values(), bkgnd.values())
    bsratio = get_ratio(   signal.values(), bkgnd.values())
    if plotPowRwgt:
        sratio  = get_ratio(bkgnd2sig.values(), signal.values())
        sbratio = get_ratio(    bkgnd.values(), signal.values())

    #default ratio
    axes[1].scatter(bins, bratio, marker='^', color='blue');
    axes[1].scatter(bins, bsratio, s=8, marker='+', color=default_cycler[0]);
    if plotPowRwgt:
        axes[1].scatter(bins, sratio, marker='v', color='orange');
        axes[1].scatter(bins, sbratio, s=4, marker='x', color='#ffc35b');
    axes[1].set_ylabel(r'$r =$ rwgt / target', fontsize=14)

    #axes[1].set_ylim(np.min([bratio, sratio]), np.max([bratio,sratio]))
    #plt.grid('both')

    feature = histo.axes[0].label
    axes[0].set_xlabel('')

    axes[0].tick_params(axis='y', which='minor', bottom=False)
    axes[0].grid(which='major', axis='both', ls='--', alpha=0.4)
    axes[0].minorticks_on()
    axes[1].minorticks_on()

    #if axes[1].get_ylim()[1]>10.:
        #axes[1].set_yscale('log')
    #else:
    if plotPowRwgt==True: interval = np.max(sratio) - np.min(sratio)
    else: interval = 0.05
    if np.max(bratio) - np.min(bratio) > interval:
        interval = np.max(bratio) - np.min(bratio)
    if 'ylimr' in kwargs and kwargs['ylimr']: 
        interval = (kwargs['ylimr'][1] - kwargs['ylimr'][0])/10
    elif interval < 0.5: interval = 0.05
    else: interval = 0.5
    interval = myround(interval, base=0.05)
    locator = pltticker.MultipleLocator(base=interval)
    axes[1].yaxis.set_minor_locator(locator)
    axes[1].grid(which='minor', axis='y', ls='--', alpha=0.3)
    axes[1].grid(which='major', axis='y', ls='--', alpha=0.4)

    if "altRatio" in kwargs:
        axes.append(fig.add_subplot(gs[2]))
        plt.subplots_adjust(hspace=0.2)
        dr_bkg = get_delta_ratio(bsratio, bratio)
        dr_sig = get_delta_ratio(sbratio, sratio)
        axes[2].scatter(bins, dr_sig,  color='orange')
        axes[2].scatter(bins, dr_bkg,  color='blue')
        
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
    if 'ylimr' in kwargs and kwargs['ylimr']:  axes[1].set_ylim(kwargs['ylimr'])
    if 'ylimr2' in kwargs: axes[2].set_ylim(kwargs['ylimr2'])

    if plotPowRwgt: axes[0].legend(ncols=2);
    else: axes[0].legend();
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

def check_selection(histo, config, makeLog=False,
          name="", runSave=False, path=None, stack=True,
          fig=None, axes=None, density=False, **kwargs
         ):
    """Plot background, signal, and reweighting"""
    if fig is None: makeit = True
    else: makeit=False

    if makeit:
        axes = []
        if "altRatio" in kwargs: 
            figsize=(7,6)
            nplots = 3
            height_ratios=[2,1,1]
        else: 
            figsize=(7,5)
            nplots = 2
            height_ratios=[3,1]
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(nplots,1, height_ratios=height_ratios)
        axes.append(fig.add_subplot(gs[0]))
        axes.append(fig.add_subplot(gs[1]))
        plt.subplots_adjust(hspace=0.2)

    #POWHEG
    bkgnd_pass  = histo[{"dataset":0, "postrwgt":0, 'sel':1}]
    bkgnd_fail  = histo[{"dataset":0, "postrwgt":0, 'sel':0}]
    bkgnd = histo[{"dataset":0, "postrwgt":0, 'sel':sum}]
    
    mplhep.histplot([bkgnd_fail,   bkgnd_pass], 
                    label=['Powheg fail sel','Powheg pass sel'], 
                    color=['#1845fb', '#964a8b'], 
                    stack=stack, histtype='fill', alpha=0.4, yerr=0, ax=axes[0]);
    #SMEFTSIM RWGT TO POWHEG
    sig2bkgnd_pass = histo[{"dataset":1, "postrwgt":1, 'sel':1}]
    sig2bkgnd_fail = histo[{"dataset":1, "postrwgt":1, 'sel':0}]
    sig2bkgnd = histo[{"dataset":1, "postrwgt":1, 'sel':sum}]
    mplhep.histplot([sig2bkgnd_fail, sig2bkgnd_pass], 
                    label=['SMEFTsim rwgt fail sel','SMEFTsim rwgt pass sel'], 
                    color=['#578dff', '#c849a9'],
                    stack=stack, yerr=0, ls='--', ax=axes[0])

    #RATIO
    feature = axes[0].get_xlabel()
    bins = bkgnd_pass.axes[0].centers

    signal = histo[{"dataset":1, "postrwgt":0, 'sel':sum}]
    bkgnd_fail_ratio = get_ratio(sig2bkgnd_fail.values(), bkgnd_fail.values() )
    bkgnd_pass_ratio = get_ratio(sig2bkgnd_pass.values(), bkgnd_pass.values())
    signal_ratio_pre  = get_ratio(signal.values(), bkgnd.values() )
    signal_ratio_post = get_ratio(sig2bkgnd.values(), bkgnd.values())

    axes[1].axhline(1., color='grey', ls='--', alpha=0.5)
    axes[1].scatter(bins, bkgnd_fail_ratio, color='#578dff', marker='v',label='fail')
    axes[1].scatter(bins, bkgnd_pass_ratio, color='#c849a9', marker='^',label='pass')
    axes[1].scatter(bins, signal_ratio_post, color='k', s=4, label='total')
    axes[1].set_ylabel('Other / Powheg')
    axes[1].legend(loc='best', ncol=3, fontsize=8)

    #RATIO OF RATIOS
    if "altRatio" in kwargs:
        axes.append(fig.add_subplot(gs[2]))
        plt.subplots_adjust(hspace=0.2)
        sig_post_pre = get_delta_ratio(signal_ratio_pre, signal_ratio_post)
        sig_post_pre_fail = get_delta_ratio(signal_ratio_pre, bkgnd_fail_ratio)
        sig_post_pre_pass = get_delta_ratio(signal_ratio_pre, bkgnd_pass_ratio)

        axes[2].axhline(1., color='grey', ls='--')
        axes[2].scatter(bins, sig_post_pre, label='total', color='k', s=12)
        axes[2].scatter(bins, sig_post_pre_fail, label='fail', color='#578dff', s=12)
        axes[2].scatter(bins, sig_post_pre_pass, label='pass', color='#c849a9', s=12)
        axes[2].set_ylabel(r'$\frac{\Delta |r-1|}{|r-1|}$', fontsize=16)
        plt.grid('both')  
        axes[2].set_xlabel(feature, fontsize=16)
    else:
        axes[1].set_xlabel(feature, fontsize=16)

    if makeLog: axes[0].set_yscale('log')
    axes[0].minorticks_on()
    axes[1].minorticks_on()
    #locator = pltticker.MultipleLocator(base=0.05)
    locator = pltticker.MultipleLocator(base=0.2)
    #axes[1].yaxis.set_minor_locator(locator)
    axes[1].grid(which='minor', axis='y', ls='--', alpha=0.3)
    axes[1].grid(which='major', axis='y', ls='--', alpha=0.4)
    
    axes[0].legend()
    axes[1].set_xlabel(feature, fontsize=12)
    axes[0].set_xlabel('')

    axes[0].set_title(f'Passing and Failing selection histograms: stacked={stack}')
    if 'xlim' in kwargs:
        axes[0].set_xlim(kwargs['xlim'])
        if "altRatio" in kwargs:
            axes[2].set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:   axes[0].set_ylim(kwargs['ylim'])
    if 'ylimr' in kwargs:  axes[1].set_ylim(kwargs['ylimr'])
    else: 
        minn = np.min(bkgnd_pass_ratio)
        if minn>np.min(bkgnd_fail_ratio): minn=np.min(bkgnd_fail_ratio)
        maxx = np.max(bkgnd_pass_ratio[bins<800])
        if maxx<np.max(bkgnd_fail_ratio[bins<800]): maxx=np.max(bkgnd_fail_ratio[bins<800])
        ylimr = [minn, maxx]
        axes[1].set_ylim(ylimr)
    if 'ylimr2' in kwargs: axes[2].set_ylim(kwargs['ylimr2'])
    
    return fig, axes