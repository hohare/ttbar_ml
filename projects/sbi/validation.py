from argparse import ArgumentParser
from numpy import linspace, median, random, ones
from os import makedirs, system
from torch import device, load
from yaml import dump, safe_load
import matplotlib.pyplot as plt
import matplotlib.table as tbl
from buildLikelihood import fullLikelihood, likelihood
from validation_plotting import histPlot, ratioPlot, parametric_table

def main(parametric):
    with open("projects/sbi/validation/"+parametric, 'r') as f:
        config = safe_load(f)
    dedicated = config['name']+config['dedicated']
    output = config['name']+config['output']

    makedirs(output, mode=0o755, exist_ok=True)
    cp_command = f'cp {parametric} {output}'
    system(cp_command)
    
    print('Loading data...')
    dataset = load(f"{config['data'].replace('train','pretrain')}/validation.p", map_location=device(config['device']), weights_only=False)

    print('Preparing dedicated likelihood...')
    dlr_obj  = likelihood(dedicated, dataset[:][0].shape[1])
    dWcs = dlr_obj.config['signalTrainingPoint']
    wcDict = {}
    for i, wc in enumerate(dlr_obj.config['wcs']):
        wcDict[wc] = dWcs[i+1]

    print('Preparing full likelihood using networks...')
    plr_obj = fullLikelihood(config, dataset[:][0])
    dlr = dlr_obj(dataset[:][0], dWcs).detach().cpu().numpy()
    plr = plr_obj(dWcs).detach().cpu().numpy()

    plr[plr < 0] = 0
    dlr[dlr < 0] = 0

    features = dataset[:][0].detach().cpu().numpy()
    fitCoefs = dataset[:][1].detach().cpu().numpy()

    print('Calculating metrics...')

    residuals = abs((dlr - plr)/dlr)
    metrics = {
        'residualMean':   float(residuals.mean()),
        'residualMin':    float(residuals.min()),
        'residualMax':    float(residuals.max()),
        'residualMedian': float(median(residuals)),
        'residualStdv':   float(residuals.std())
    }
    
    makedirs(f'{output}/noNorm',  mode=0o755, exist_ok=True)
    makedirs(f'{output}/density', mode=0o755, exist_ok=True)

    data, rows, cols, colors = parametric_table(dedicated, config)
    for kinematic, params in config['features'].items():
        print(f'- making plots for kinematic {kinematic}')
        bins = linspace(params['min'], params['max'], params['nbins'])
        fig, ax = ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict,
                  xlabel=params['label'])
        tbl.table(ax[0], cellText=data, rowLabels=rows, colLabels=cols,
                 cellLoc='center', loc='top', cellColours=colors, colWidths=ones(len(cols))*0.1,)
        fig.savefig(f'{output}/noNorm/{kinematic}.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        fig, ax = ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict,
                  xlabel=params['label'], plotLog=True)
        tbl.table(ax[0], cellText=data, rowLabels=rows, colLabels=cols,
                 cellLoc='center', loc='top', cellColours=colors, colWidths=ones(len(cols))*0.1,)
        fig.savefig(f'{output}/noNorm/{kinematic}_log.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        fig, ax = ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict,
                  xlabel=params['label'], density=True)
        tbl.table(ax[0], cellText=data, rowLabels=rows, colLabels=cols,
                 cellLoc='center', loc='top', cellColours=colors, colWidths=ones(len(cols))*0.1,)
        fig.savefig(f'{output}/density/{kinematic}.png', bbox_inches='tight')
        plt.show()
        plt.close()
        
        fig, ax = ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict,
                  xlabel=params['label'], density=True, plotLog=True)
        tbl.table(ax[0], cellText=data, rowLabels=rows, colLabels=cols,
                 cellLoc='center', loc='top', cellColours=colors, colWidths=ones(len(cols))*0.1,)
        fig.savefig(f'{output}/density/{kinematic}_log.png', bbox_inches='tight')
        plt.show()
        plt.close()

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('parametric', help = 'configuration yml file used for parametric likelihood')
    args = parser.parse_args()
    main(args.parametric)
