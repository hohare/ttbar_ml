import os
from argparse import ArgumentParser
from tqdm import tqdm
from yaml import safe_load

from torch import cuda, load, optim, save, device
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from training.EarlyStopper import EarlyStopper

def main(project, config):
    #Check for GPU availability and fall back on CPU if needed
    if config['device'] != 'cpu' and not cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        config['device'] = 'cpu'
    if config['device'] == 'cuda': print('Will run using cuda!')

    os.makedirs(config['name']+"/training", mode=0o755, exist_ok=True)

    print('Loading data...')
    if project=="sbi":
        # Load dataset features and structure coefficients
        from projects.sbi.net import Model
        import training.weight_manager as wgtMan
        import projects.sbi.network_plotting as myplt

        train = load(f'{config["data"]}/train.p', map_location=device(config['device']), weights_only=False)
        test  = load(f'{config["data"]}/test.p', map_location=device(config['device']), weights_only=False)
        # Normalize features
        train[:][0][:] = (train[:][0] - train[:][0].mean(0))/train[:][0].std(0)
        test[:][0][:]  = (test[:][0] - test[:][0].mean(0))/test[:][0].std(0)
        # Change dataset structure coefficients to background and signal weights
        train, test = wgtMan.calculate_weights(train, test, config)

        model = Model(nFeatures=train[:][0].shape[1],device=config['device'], config=config['network'])
    elif project=="dctr":
        from projects.dctr.net import Model
        import training.network_plotting as myplt
        
        train = load(config["traindata"], map_location=device(config['device']), weights_only=False)
        test  = load(config["testdata"], map_location=device(config['device']), weights_only=False)
        
        model = Model(nFeatures=train[:][0].shape[1], device=config['device'], network=config['network'])

    # Batch the dataset
    dataset_batches = DataLoader(train, batch_size=config['batchSize'], shuffle=True)#, num_workers=4)

    # Prepare for training
    optimizer = optim.Adam(model.net.parameters(), lr=config['learningRate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config['factor'], patience=config['patience'])
    stopper   = EarlyStopper(tolerance=config['patience']+1, delta=config['delta'])
    trainLoss = [model.loss(train[:][0], train[:][1], train[:][2]).item()]
    testLoss  = [model.loss(test[:][0],  test[:][1],  test[:][2]).item()]

    print('Training...')
    for epoch in tqdm(range(config['epochs'])):
        if epoch%10==0: 
            myplt.plot_losses(testLoss, trainLoss, config['name'], epoch)
            myplt.plot_network(model.net, test, config['name'], epoch)
        totloss = 0 #sometime train tensor is too big to calc on whole
        for features, weights, weights_or_label in dataset_batches:
            optimizer.zero_grad()
            loss = model.loss(features, weights, weights_or_label)
            totloss += loss
            loss.backward()
            optimizer.step()

        trainLoss.append((totloss/len(dataset_batches)).detach().cpu().numpy())
        #trainLoss.append(model.loss(train[:][0], train[:][1], train[:][2]).item())
        testLoss.append(model.loss(test[:][0], test[:][1], test[:][2]).item())
        scheduler.step(testLoss[epoch])
        stopper(testLoss[-1])
        if stopper.stop_early:
            print(f'Stopping early after {epoch} epochs')
            break

    print("Plotting network results...")
    myplt.plot_network_end(model.net, test, train, testLoss, trainLoss, config['name'], epoch+1)
    # Save network final results
    print("Saving network...")
    net = model.net.to('cpu') #Recommended by pytorch
    save(net, config['name']+'/network.p')
    save(net.state_dict(), config['name']+'/networkStateDict.p')

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('project', help = 'which project is this for?')
    parser.add_argument('config', help = 'configuration yml file used for training')
    args = parser.parse_args()
    
    #Load the configuration options and build the WC lists
    #with open("/uscms_data/d3/honor/Outputs_sbi/training/ctq8_1p22_basic/"+parser.parse_args().config, 'r') as f:
    with open(args.config) as f:
        config = safe_load(f)
    main(args.project, config)
    