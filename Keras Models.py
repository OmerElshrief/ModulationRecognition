from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout

#from keras.layers import Embedding, Conv1D, Convolution1D, MaxPooling1D
#from keras.layers import Concatenate, SpatialDropout1D, Reshape

from keras.regularizers import l2

from keras.optimizers import Adam

from keras.callbacks import TensorBoard
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from keras.layers import LSTM, TimeDistributed
from keras.models import model_from_json

basic_model = Sequential()

basic_model.add(Conv2D(64, kernel_size=(1, 3), padding = 'same', activation= 'relu', input_shape=(2, 128, 1)))
basic_model.add(Dropout(0.5))
basic_model.add(Conv2D(16, kernel_size=(2, 3), padding = 'same', activation= 'relu'))
#basic_model.add(Dropout(0.5))
basic_model.add(Flatten())
basic_model.add(Dense(128, activation='relu'))
basic_model.add(Dropout(0.5))
basic_model.add(Dense(10, activation='softmax'))

basic_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics = ['accuracy'])
basic_model.summary()

## ----------------------------------------------------------------------- ##

deep_model = Sequential()

deep_model.add(Conv2D(256, kernel_size=(1, 3), padding = 'same', activation= 'relu', input_shape=(2, 128, 1)))
deep_model.add(Dropout(0.6))
deep_model.add(Conv2D(256, kernel_size=(2, 3), padding = 'same', activation= 'relu'))
deep_model.add(Dropout(0.6))
deep_model.add(Conv2D(80, kernel_size=(1, 3), padding = 'same', activation= 'relu'))
deep_model.add(Dropout(0.6))
deep_model.add(Conv2D(80, kernel_size=(1, 3), padding = 'same', activation= 'relu'))
deep_model.add(Dropout(0.6))
deep_model.add(Flatten())
deep_model.add(Dense(128, activation='relu'))
deep_model.add(Dropout(0.5))
deep_model.add(Dense(10, activation='softmax'))

deep_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics = ['accuracy'])
deep_model.summary()

## ------------------------------------------------------------------------- ##

lstm_model = Sequential()

lstm_model.add(Conv2D(128, kernel_size=(1, 3), padding = 'same', activation= 'relu', input_shape=(2, 128, 1)))
lstm_model.add(Dropout(0.6))
lstm_model.add(Conv2D(64, kernel_size=(2, 3), padding = 'same', activation= 'relu'))
lstm_model.add(Dropout(0.6))
lstm_model.add(Conv2D(32, kernel_size=(1, 3), padding = 'same', activation= 'relu'))
lstm_model.add(Dropout(0.6))
lstm_model.add(Conv2D(16, kernel_size=(2, 3), padding = 'same', activation= 'relu'))
lstm_model.add(Dropout(0.6))
lstm_model.add(TimeDistributed(Flatten()))
lstm_model.add(LSTM(64))
lstm_model.add(Dense(128, activation='relu'))
lstm_model.add(Dropout(0.6))
lstm_model.add(Dense(10, activation='softmax'))

lstm_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics = ['accuracy'])
lstm_model.summary()



## ----------------------------------------------------------------------- ##

# LSTM-Resnet deep Model

from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Add
# Define the residual block as a new layer
class Residual(Layer):
    def __init__(self, channels_in,kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

    def call(self, x):
        # the residual block using Keras functional API
        first_layer =   Activation("linear", trainable=False)(x)
        x =             Conv2D( self.channels_in,
                                self.kernel,
                                padding="same")(first_layer)
        x =             Activation("relu")(x)
        x =             Conv2D( self.channels_in,
                                self.kernel,
                                padding="same")(x)
        residual =      Add()([x, first_layer])
        x =             Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
 

lstmResnet_model = Sequential()

lstmResnet_model.add(Conv2D(128, kernel_size=(1, 3), padding = 'same', activation= 'relu', input_shape=(2, 128, 1)))
lstmResnet_model.add(Dropout(0.6))
#.add(Residual(128,(1,3)))
lstmResnet_model.add(Conv2D(64, kernel_size=(2, 3), padding = 'same', activation= 'relu'))
lstmResnet_model.add(Dropout(0.6))

lstmResnet_model.add(Residual(64,(2,3)))

lstmResnet_model.add(Conv2D(32, kernel_size=(1, 3), padding = 'same', activation= 'relu'))
lstmResnet_model.add(Dropout(0.6))
lstmResnet_model.add(Residual(32,(1,3)))
lstmResnet_model.add(Residual(32,(2,3)))
lstmResnet_model.add(Conv2D(16, kernel_size=(1, 3), padding = 'same', activation= 'relu'))
lstmResnet_model.add(Dropout(0.6))

lstmResnet_model.add(TimeDistributed(Flatten()))
lstmResnet_model.add(LSTM(64))

lstmResnet_model.add(Dense(128, activation='relu'))
lstmResnet_model.add(Dropout(0.6))
lstmResnet_model.add(Dense(10, activation='softmax'))

lstmResnet_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics = ['accuracy'])
lstmResnet_model.summary()
