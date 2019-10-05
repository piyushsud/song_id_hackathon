import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, optimizers, models
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
import sklearn


class dnn_ID_classifier:
    def __init__(self, num_hidden_layers=1, hidden_layer_size=10, dropout=0, 
                 loss='sparse_categorical_crossentropy', learning_rate=.01, 
                 epochs=20, optimizer='sgd', batch_size=128, verbose=1):
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.norm_val = 1
      
    def build_model(self, input_size, output_size):
        m_input = layers.Input(input_size)
#        model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])
        model = models.Sequential()
        model.add(Dense(1024,input_shape=input_size))
        model.add(Dense(512,input_shape=input_size))
        model.add(Dense(output_size,input_shape=input_size))
        
        model.compile(self.optimizer, self.loss, metrics=['accuracy'])
        
    def fit(self, X, y):
        self.norm_val = np.max(np.abs(X), axis=0)
        X = X/self.norm_val
        history = model.fit(X, y, self.epochs, self.batch_size)
      
    def evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
        (test_loss, test_accuracy) = model.evaluate(X_test, y_test)
        
    def predict(self,X):
        self.norm_val = np.max(np.abs(X),axis=0)
        X=X/self.norm_val
        proba = model.predict(X)
        return np.argmax(proba, axis=1)
        