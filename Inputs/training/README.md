# Training configuration file

- `name`: path to place the training output
- `signalSample`: list of files to concatenate to comprise the signal sample
- `signalMetadata`: a .json containing sum of weights necessary for xsec signal normalization
- `backgroundSample`: list of files to concatenate to comprise the background sample
- `backgroundMetadata`: a .json containing sum of weights necessary for background xsec normalization
- `device`: whether to run on cpu or gpus (cuda)
- `network`: which network structure to use
- `features`: list of features to be extracted from dataframe (if dataframe exists) otherwise is the order of features in the feature tensor
- `batchSize`: nEvents per batch for training
- `epochs`: nIteration to train over the dataset
- `normalization`: type of weight normalization, if any
- `learningRate`: rate of decrease for training optimizer
- `patience`: nIterations for scheduler to wait before reducing the learning rate
- `factor`: how much the scheduler should reduce the learning rate
- `delta`: minimum required decrease in loss each iteration to avoid early stopping