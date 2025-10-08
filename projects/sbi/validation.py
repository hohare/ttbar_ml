from argparse import ArgumentParser
from numpy import linspace, median, random
from os import makedirs, system
from training.buildLikelihood import fullLikelihood, likelihood
from training.plots import histPlot, ratioPlot
from torch import device, load
from yaml import dump, safe_load

def main(parametric):
    with open(parametric, 'r') as f:
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
    
    #return plr, dlr
    residuals = abs((dlr - plr)/dlr)

    print('Calculating metrics...')
    
    makedirs(f'{output}/noNorm',  mode=0o755, exist_ok=True)
    makedirs(f'{output}/density', mode=0o755, exist_ok=True)
    
    for kinematic, params in config['features'].items():
        print(f'- making plots for kinematic {kinematic}')
        bins = linspace(params['min'], params['max'], params['nbins'])
        ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict, 
                  outname=f'{output}/noNorm/{kinematic}.png', 
                  xlabel=params['label'], showNoWeights=True,)
        ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict, 
                  outname=f'{output}/noNorm/{kinematic}_log.png', 
                  xlabel=params['label'], showNoWeights=True, plotLog=True)
        ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict, 
                  outname=f'{output}/density/{kinematic}.png', 
                  xlabel=params['label'], showNoWeights=True, density=True)
        ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict, 
                  outname=f'{output}/density/{kinematic}_log.png', 
                  xlabel=params['label'], showNoWeights=True, density=True, plotLog=True)

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('parametric', help = 'configuration yml file used for parametric likelihood')
    args = parser.parse_args()
    main(args.parametric)
