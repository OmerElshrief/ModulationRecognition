## Loading - Splitting and Shuffling the Data based on SNR

import _pickle as pkl
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

#  You will need to seperately download or generate this file
Xd =  pkl.load(open("drive/My Drive/RML2016.10b.dat",'rb'),encoding='latin1')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# The data needs to be shuffled 
np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.7)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    
    lyy = list(yy)
    yy1 = np.zeros([len(lyy), 10])
    yy1[np.arange(len(lyy)),lyy] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
# Splitting the Data

X = np.reshape(X_train, (-1, 2, 128, 1))
train_X, val_X, train_y, val_y = train_test_split(X, Y_train, test_size = 0.3, random_state = 1)