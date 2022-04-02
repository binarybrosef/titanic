##--Binary classification model predicting survival outcomes for the Kaggle Titanic challenge
##--This model comprises 2 hidden layers, 14 hidden units per layer, and a sigmoid activation
##--function at the output layer. This model is configured to train on training and test sets
##--that are separately normalized/filled in where missing values exist to avert leakage.
##--At 50 epochs, this model achieves a score of 0.76315.


from preprocessing import x_train, y_train, x_test
import tensorflow as tf
import numpy as np  														
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from os import getcwd


NAME = '2-hidden_layers-14-hidden_units-Sigmoid-True'

#Create input layer
model = Sequential(name=NAME)
model.add(Dense(14, input_shape=(16,)))
model.add(Activation('relu'))

#Add two hidden layers
for i in range(2):
	model.add(Dense(14))
	model.add(Activation('relu'))

#Add output layer
model.add(Dense(1))

#Add sigmoid activation function to output layer
model.add(Activation('sigmoid'))
logits = False

model.compile(
	loss=tf.keras.losses.BinaryCrossentropy(from_logits=logits),
	optimizer="adam",
	metrics=["accuracy"],
)

#shuffle=True by default
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)


#Save model and weights
model.save(getcwd() + '\\model')
model.save_weights(getcwd() + '\\weights')

#Predict on test set, save predictions in .csv format as integers
predictions = np.round(model(x_test))
np.savetxt('predictions.csv', predictions, fmt='%.i', delimiter=',')