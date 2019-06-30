## Model build to build anymodel you want, just choose the Parameters
class ModelBuilder:
  
  def__init__(self, filter_sizes, filter_shapes, dropout_probs,
              rnn = False, rnn_cells = 50,
              loss = 'categorical_crossentropy', optimizer = 'Adam'):
    self.conv_layers_cnt = filter_sizes.size()
    self.filter_sizes = filter_sizes
    self.filter_shapes = filter_shapes
    self.rnn = rnn
    self.rnn_cells = rnn_cells
    self.model = model
    
    
  def build_model():
    self.model = Sequential()
    self.model.add(Conv2D(self.filter_sizes[0], kernel_size=self.filter_shapes[0], padding = 'same', activation= 'relu', input_shape=(2, 128, 1)))
    self.model.add(Dropout(self.dropout_probs[0]))
    
    for f in range(1, self.conv_layers_cnt):
      self.model.add(Conv2D(self.filter_sizes[f], kernel_size=self.filter_sizes[f], padding = 'same', activation= 'relu'))
      self.model.add(Dropout(self.dropout_probs[f]))
      
    if self.lstm:
      self.model.add(TimeDistributed(Flatten))
      se;f.model.add(GRU(self.rnn_cells))
      
    else:
      self.model.add(Flatten)
      
    self.model.add(Dense(128, activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(10, activation='softmax'))  
    
    self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics = ['accuracy'])
    self.model.summary()
    return self.model