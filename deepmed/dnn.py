import numpy as np
import random
import pandas as pd
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l1



def make_dnn(train_ybin,input_dim,lambda_value,layers,units):
    model = Sequential()
    i=0
    while i<layers:
        if i==0:
            n_input=input_dim
            lambda_l1=l1(lambda_value)
            units_i = units
        else:
            n_input = units
            lambda_l1=None
            units_i = units
        model.add(Dense(units = units_i,activation = 'relu', input_shape = (n_input,),
                        kernel_regularizer = lambda_l1,
                        kernel_initializer = GlorotUniform(seed=1),dtype='float64'))
        model.add(BatchNormalization())
        i+=1
    if train_ybin==1:
        model.add(Dense(units = 1,activation = 'sigmoid', kernel_initializer = GlorotUniform(seed=1),dtype='float64'))
        model.compile(optimizer=tf.keras.optimizers.Adam(),loss = 'binary_crossentropy',metrics = ['binary_crossentropy'])
    else:
        model.add(Dense(units = 1, kernel_initializer = GlorotUniform(seed=1),dtype='float64'))
        model.compile(optimizer=tf.keras.optimizers.Adam(),loss = 'mean_squared_error',metrics = ['mean_squared_error'])
                        
        
    return model

def dnn(y, x, ytest, xtest, hyper):
    train_ybin=1*((len(np.unique(y))==2) & (min(y)==0) & (max(y)==1))
    l1_val=hyper[0]
    layers=int(hyper[1])
    units=int(hyper[2])
    epochs=math.ceil(hyper[3])
    batch_size=int(hyper[4])
    input_dim = x.shape[1]

    NN = make_dnn(train_ybin,input_dim,l1_val,layers,units)
    y = y.reshape(-1, 1)  # Reshape to (batch_size, 1)
    ytest = ytest.reshape(-1, 1)    # Do the same for the test set if applicable

    # if train_ybin==1:
    #     NN.compile(optimizer='adam',loss = 'binary_crossentropy',metrics = ['binary_crossentropy'])
    # else:
    #     NN.compile(optimizer='adam',loss = "mean_squared_error",metrics = ['mean_squared_error'])
    NNfit = NN.fit(x, y, epochs =epochs, batch_size=batch_size, verbose = False,validation_data=(xtest, ytest))   
        
    ypred = NN(xtest).numpy().flatten()
    val_loss_all = NNfit.history["val_loss"]
    loss = min(val_loss_all)
    epoch_opt = np.argmin(val_loss_all)
    if epoch_opt==0:
        epoch_opt=1
    return loss,ypred,epoch_opt   
    
    
    
