
## Tuning details

basic_model_params{
    'filter_sizes': [64, 16],
    'filter_shapes': [(1, 3), (2, 3)],
    'dropout_probs': [0.6, 0.6],
    'rnn' = False,
    'logger': 'basicCNN'
}

deep_model_params{
    'filter_sizes': [128, 64, 32, 16],
    'filter_shapes': [(1, 3), (2, 3), (1, 3), (1, 3)],
    'dropout_probs': [0.6, 0.6, 0.6, 0.6],
    'rnn' = False,
    'logger': 'deepCNN'
}

lstm_model_params{
    'filter_sizes': [128, 64, 32, 16],
    'filter_shapes': [(1, 3), (2, 3), (1, 3), (1, 3)],
    'dropout_probs': [0.6, 0.6, 0.6, 0.6],
    'rnn' = True, 
    'logger': 'CLDNN'
}


## Function to save the Model while training...
def save_model(model, path, arch = True, weights = False):
  # serialize model to JSON
  model_json = model.to_json()
  with open(os.path.join(path, 'arch.json'), "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(path, 'weights.h5'))
    print("model Weights are saved")

def load_model(path):
    son_file = open((path+'/arch.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(path + '/weights.h5')
    print("Loaded model from disk")
    model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics = ['accuracy'])
    model.summary()
    return model

## -------------------------------------------------------------------------------##
# Preparing Training Data 

## for Keras Model 
X = np.reshape(data, (1200000, 2, 128, 1))

X_train, X_test, y_train, y_test = train_test_split(X, encoded, test_size = 0.3, random_state = 1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

enc = OneHotEncoder()
k_train = enc.fit_transform(y_train.reshape(-1, 1))

## Splitting training data into train and Validation 
train_X, val_X, train_y, val_y = train_test_split(X_train, k_train, test_size = 0.3, random_state = 1)

## Training LSTM Model 
# You may change the path as you desire
path = "drive/My Drive/models/keras/lstm_weights1new.h5"
cp_callback = ModelCheckpoint(path ,
                              monitor = 'val_acc',
                              save_best_only = True,
                              save_weights_only=True,
                              mode = 'max',
                              verbose=1)

##early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=2, verbose=0, mode='max')

history = lstm_model.fit(x=train_X, y=train_y,
          batch_size= 1024, epochs= 50,
          verbose=2,
          callbacks = [cp_callback],
          validation_data=(val_X, val_y)
          )

his.append(history)


# Training results 
history = his[0]
history2 = his[1]
# history may differ according to how many time have you paused training.
plt.figure()
plt.title('Training performance')
plt.plot(no_epochs, history.history['loss']+history2.history['loss'], label='train loss+error')
plt.plot(no_epochs, history.history['val_loss']+history2.history['val_loss'], label='val_error')
plt.legend()

# Testing
# Overall results
X_test=X_test.reshape(-1,2,128,1)
score = lstm_model.evaluate(X_test, Y_test,  verbose=0, batch_size=1024)
print('Total overall score => Loss: {:.4f}   Over-All Accuracy: {:.0f}%'.format(score[0],100*score[1]))

## SNR testing and confusion matrix plotting
# Plot confusion matrix
acc = {}
classes = [0,1,2,3,4,5,6,7,8,9]
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    
    
    test_Y_i= np.array([np.argmax(x) for x in test_Y_i])
    #print(test_Y_i)
   
    # estimate classes
    test_X_i = (test_X_i).reshape(test_X_i.shape[0] , 2,128,1)
    test_Y_i_hat = lstm_model.predict(test_X_i)
    
    
    #print(test_Y_i_hat)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = test_Y_i[i]
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("SNR: {} - Accuracy: {:.0f}%   no. of test Signals: {} ".format(snr ,100* cor / (cor+ncor),test_X_i.shape[0]))
    acc[snr] = 1.0*cor/(cor+ncor)