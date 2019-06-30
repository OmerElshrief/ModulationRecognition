# ModulationRecognition

## DEEPSIG DATASET: RADIOML 2016.10A
A synthetic dataset, generated with GNU Radio, consisting of 11 modulations (8 digital and 3 analog) at varying signal-to-noise ratios.
The file is formatted as a "pickle" file which can be open for example in python by using cPickle.load(...)
Dataset Download: http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2

## Models 
We used five different models and compared the results of each model.
-Basic NN Model 
-Deep CNN Model
-ResNet Model
-LSTM model
-LSTM-RestNet combined Model

Best Results was for LSTM-ResNet Model which was Overall accuracy 62% , 89% for High SNR

