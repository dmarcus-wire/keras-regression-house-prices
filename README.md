# Keras Regression

## Goal
- Train a Keras neural network to predict house prices based on categorical and numerical attributes such as the number of bedrooms/bathrooms, square footage, zip code, etc.
- Regression will be able to predict an exact dollar amount, such as “The estimated price of this house is $489,121”.

## Model
Our Keras regression architecture. 
The input to the network is a datapoint including a home’s # Bedrooms, # Bathrooms, Area/square footage, and zip code. 
The output of the network is a single neuron with a linear activation function. Linear activation allows the neuron to output the predicted price of the home.

## Dataset
source: https://github.com/emanhamed/Houses-dataset

### attributes
- No. of Bedrooms
- No. of Bathrooms
- No. Area (sqft)
- zipcode
