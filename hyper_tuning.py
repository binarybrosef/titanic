##--Hyperparameter tuning for a binary classification model predicting survival outcomes--
##--for the Kaggle Titanic challenge. 60 different models, consisting of different numbers of
##--hidden layers, different numbers of hidden units (same number of hidden units per hidden layer)
##--and different output layer activation functions, are computed. Tensorboard is used to visualize
##--accuracy and loss, and to choose the best performing model according to these metrics. 
##--Best performing model appears to comprise 2 hidden layers, 14 hidden units per layer, and
##--a sigmoid activation function at the output layer.


from preprocessing import x_train, y_train			#Import training set and labels from preprocessing.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard

layer_nums = [1, 2, 3]								#Number of hidden layers
unit_nums = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]	#Number of units in each hidden layer
sigmoids = [True, False]							#Sigmoid activation/no activation at output layer


for layer_num in layer_nums:						#Try different numbers of hidden layers
	for unit_num in unit_nums:						#Try different numbers of hidden units per hidden layer
		for sigmoid in sigmoids:					#Try using sigmoid activation at output layer versus no activation

			#Assign unique name to each model
			NAME = f'{layer_num}-hidden_layers-{unit_num}-hidden_units-Sigmoid-{sigmoid}'

			#Create input layer
			model = Sequential(name=NAME)
			model.add(Dense(unit_num, input_shape=(16,)))
			model.add(Activation('relu'))

			#Add hidden layer(s)
			for i in range(layer_num):
				model.add(Dense(unit_num))
				model.add(Activation('relu'))

			#Add output layer
			model.add(Dense(1))
			
			#If sigmoid enabled, add sigmoid activation function to output layer
			if sigmoid:
				model.add(Activation('sigmoid'))
				logits = False
			elif sigmoid == False:
				logits = True

			tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

			model.compile(
				loss=tf.keras.losses.BinaryCrossentropy(from_logits=logits),
				optimizer="adam",
				metrics=["accuracy"],
			)

			#shuffle=True by default
			history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[tensorboard])