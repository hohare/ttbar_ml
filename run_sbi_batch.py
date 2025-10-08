import os
import yaml
from argparse import ArgumentParser
from run_training import main as train_main

def main(batchconfig, baseconfig):
    if batchconfig['wcs'] != baseconfig['wcs']:
        print('WC lists do not match! Check your WC ordering!')
        return 1
    
    for trainpt in batchconfig['signalTrainingPoint']:
        output = os.path.join(batchconfig['name'].replace("pre",""),trainpt)
        baseconfig['signalTrainingPoint'] = batchconfig['signalTrainingPoint'][trainpt]
        baseconfig['data'] = batchconfig['data']
        baseconfig['name'] = output
    
        os.makedirs(f'{output}', mode=0o755, exist_ok=True)
        trainfile = f'{output}/training.yml'
        with open(trainfile, 'w') as f:
            yaml.dump(baseconfig, f)

        print(f'RUNNING TRAINING FOR {trainpt}')
        #try: 
        res = train_main("sbi", baseconfig)
        #except:
        #    print('oopsie')


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('-t', '--batch',
                        default = 'Inputs/training/batch.yml',
                        help = 'configuration yml file used for train points')
    parser.add_argument('-b', '--base', 
                        default = 'Inputs/training/base.yml',
                        help = 'configuration yml file used for training')
    args = parser.parse_args()

    with open(args.batch, 'r') as f:
        batchconfig = yaml.safe_load(f)
    with open(args.base, 'r') as f:
        baseconfig = yaml.safe_load(f)
    
    main(batchconfig, baseconfig)