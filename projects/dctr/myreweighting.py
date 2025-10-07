import hist
import yaml
import torch
from projects.dctr.net import Net

def load_network(config):
    model = Net(len(config['features']), 'cpu', config['network'])
    name = config['name']
    model.load_state_dict(torch.load(f'{name}/networkStateDict.p'))#, map_location=device('cpu')))
    model.eval()
    return model

def compute(net, features):
    s  = net(features)
    ratio = (s/(1-s)).flatten()
    return ratio

def do_val_weighting(net, dataloader, config, doFeatures, runSave=False, makeLog=False, **kwargs):
    all_histos = make_clean_histograms(doFeatures, config)
    all_histos['weights'] = hist.Hist(
        hist.axis.Regular(100, 0., 50., name='wgt', label='wgt'),
        hist.axis.IntCategory([0,1], name='dataset', label='dataset'),
    )
    all_histos['s'] = hist.Hist(
        hist.axis.Regular(50, 0., 1., name='output', label='output'),
        hist.axis.IntCategory([0,1], name='dataset', label='dataset'),
    )
    all_histos['factor'] = hist.Hist(
        hist.axis.Regular(50, 0., 10., name='factor', label='factor'),
        hist.axis.IntCategory([0,1], name='dataset', label='dataset'),
    )

    means = torch.tensor(config['fmeans']) # device=config['device'])
    stdvs = torch.tensor(config['fstds'])  # device=config['device'])
    
    for test_features, test_weights, test_targets in dataloader:
        s  = net(test_features).flatten()
        ratios = compute(net, test_features).flatten()

        test_weights = torch.where(test_targets==1., 
                                   test_weights*config['test_wgtmeans'][1],
                                   test_weights*config['test_wgtmeans'][0]
                                  )
        realweights = torch.where(test_targets==1., test_weights/ratios, test_weights*ratios)
        totfeatures = (test_features * stdvs + means)

        targets = test_targets.detach().cpu().numpy().astype(int)
        for feature in config['features']:
            # Find feature column in tensor
            fidx = config['features'].index(feature)
            fillfeature = totfeatures[:,fidx].detach().cpu().numpy()
            # Fill feature hist
            all_histos[feature].fill(
                feature = fillfeature,
                dataset = targets,
                postrwgt = 0,
                weight = test_weights.detach().cpu().numpy(),
                factor = ratios.detach().cpu().numpy()
            )
            all_histos[feature].fill(
                feature = fillfeature,
                dataset = targets,
                postrwgt = 1,
                weight = realweights.detach().cpu().numpy(),
                factor = ratios.detach().cpu().numpy()
            )

        all_histos['weights'].fill(
            wgt = test_weights.detach().cpu().numpy(),
            dataset = targets
        )
        all_histos['s'].fill(
            output = s.detach().cpu().numpy(),
            dataset = targets
        )
        all_histos['factor'].fill(
            factor = ratios.detach().cpu().numpy(),
            dataset = targets
        )
        
    return all_histos

def make_clean_histograms(doFeatures, config):
    if doFeatures: hlist=config['features']
    else: hlist=config['variables']
    with open('/home/honor/uscmsdata/ttbareft/ttbar_ml/Inputs/histogram_settings.yml') as f:
        hset = yaml.safe_load(f)
        
    hist_dict = {}
    for variable in hlist:
        if variable in hset.keys():
            hist_dict[variable] = hist.Hist(
                hist.axis.Regular(hset[variable]['nbins'], hset[variable]['min'], hset[variable]['max'], name='feature', label=variable),
                hist.axis.IntCategory([0,1], name='dataset', label='dataset'),
                hist.axis.IntCategory([0,1], name='postrwgt', label='postrwgt'),
                #hist.axis.IntCategory([0,1], name='cuts', label='with cuts')
                hist.axis.Regular(50, 0., 10., name="factor", label="factor"),
                storage='Weight'
            )
        else:
            print(f'Histogram for variable {variable} not defined.')
    return hist_dict