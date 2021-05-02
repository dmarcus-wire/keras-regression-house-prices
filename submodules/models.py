# Multi-Layer Perceptron architecture implementation

# import the necessary packages
from tensorflow.keras.models import Sequential #
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# create multilayer perceptron
# dim is length of feature vector from line 60-61 in
# datasets.py
# regress boolean True if performing regression
def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	# 8 nodes fully connected with a relu activation
	model.add(Dense(8, input_dim=dim, activation="relu"))
	# 4 nodes fully connected with a relu activation
	model.add(Dense(4, activation="relu"))

	# check to see if the regression node should be added
	if regress:
		# 1 node fully connected with a linear activation
		# corresponds with target value we want to predict $$$
		# remember all house prices will be between 0 and 1
		# relu could result in negative values
		# linear activation aligns bes
		model.add(Dense(1, activation="linear"))

	# return our model
	return model