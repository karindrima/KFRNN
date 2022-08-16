#!/usr/bin/env python
# coding: utf-8

# In[50]:


import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the training set
dataset_train = pd.read_csv('~/IEEE14.csv')
training_set = dataset_train.iloc[:,2:].values


# In[57]:


training_set.shape, training_set


# In[93]:


from torch import nn
import torch
rnn = nn.LSTM(11, 100).to(torch.double)
train = torch.tensor(training_set)
train.to(torch.double)
print(train.size(), train.dtype, train)
print(rnn)
out = rnn(train)
print(out[0].size(), out[1][0].size(), out[1][1].size())
print(out)


# In[ ]:





# In[94]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[96]:


#creating a data structure with 60 timesteps and 1 
X_train = []
Y_train = []
for i in range (50, 2000):
    X_train.append(training_set_scaled[i-50:i, 0])
    Y_train.append(training_set_scaled[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

#Reshaping 
X_train = np.reshape((X_train.shape[0], X_train.shape[1],1))


# In[20]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')


# In[21]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[22]:


# Initialising the RNN
regressor = Sequential()


# In[97]:


#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1,1])))
regressor.add(Dropout(0.2))

#Adding a second LSTM layer and some dropout regulation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a third LSTM layer and some dropout regulation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a third LSTM layer and some dropout regulation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[25]:


#output Layer
regressor.add(Dense(units = 1))


# In[27]:


#compiling RNN
regressor.compile(optimizer = 'adam', loss= 'mean_squared_error')


# In[98]:


#Fitting the RNN to the Training set
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)


# In[100]:


#Loading the test data
dataset_test = pd.read.csv('~/IEEE14.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# In[101]:


# Getting the predicted value
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 50:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_value = regressor.predict(X_test)
predicted_value = sc.inverse_transform(predicted_stock_price)


# In[ ]:




