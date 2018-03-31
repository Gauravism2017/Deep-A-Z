###RNN
import time
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

t1 =time.time()

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Dataset

training_set = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = training_set.iloc[:,1:2].values

#Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#Getting the input and output
x_train = training_set[0:1257]
y_train = training_set[1:1258]


#Reshaping
x_train = np.reshape(x_train, (1257, 1, 1))

#Importing keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Intializing RNN
regressor = Sequential()

#Adding layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

#Adding Output layer

regressor.add(Dense(units = 1))

#Compiling RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the training set

regressor.fit(x_train, y_train, batch_size = 32, epochs = 200)


test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values


#Getting the predicted stock prices
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

t2 = time.time()
print(t2 - t1)

#Visualing the result
plt.plot(real_stock_price, color = 'red', label = 'Real_Google_Stock_price')
plt.plot(predicted_stock_price, color = 'blue', label = 'predictedStock price')
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



