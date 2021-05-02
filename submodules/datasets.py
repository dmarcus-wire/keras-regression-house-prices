# script for loading the numerical/categorical data from the dataset

# import necessary packages
from sklearn.preprocessing import LabelBinarizer # label encoder for zip codes
from sklearn.preprocessing import MinMaxScaler # find max/min of each column and scale to 0 or 1 range
import pandas as pd # load csv file and parse
import numpy as np # numerical array
import glob
import cv2
import os

# initialize load-house-attributes method
def load_house_attributes(inputPath): # load .txt from disk
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	# load from disk, separate with space, no headers, name of columns
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

	# class imbalance on the zip codes, so need to remove some classes in the future
	# determine (1) the unique zip codes
	zipcodes = df["zipcode"].value_counts().keys().tolist()
	# and (2) the number of data points with each zip code
	counts = df["zipcode"].value_counts().tolist()

	# loop over each of the unique zip codes and their corresponding
	# count = contain total houses in a zipcode
	for (zipcode, count) in zip(zipcodes, counts):
		# the zip code counts for our housing dataset is *extremely*
		# unbalanced (some only having 1 or 2 houses per zip code)
		# so let's sanitize our data by removing any houses with less
		# than 25 houses per zip code
		if count < 25:
			idxs = df[df["zipcode"] == zipcode].index
			# drop 'em
			df.drop(idxs, inplace=True)

	# return the data frame
	return df

def process_house_attributes(df, train, test):
	# initialize the column names of the continuous data
	continuous = ["bedrooms", "bathrooms", "area"]

	# performing min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler() # see wiki for definition
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])

	# one-hot encode the zip code categorical data (by definition of
	# one-hot encoding, all output features are now in the range [0, 1])
	zipBinarizer = LabelBinarizer().fit(df["zipcode"])
	trainCategorical = zipBinarizer.transform(train["zipcode"])
	testCategorical = zipBinarizer.transform(test["zipcode"])

	# construct our training and testing data points by concatenating
	# the categorical feature vector with the continuous features
	# these will get passed into the nueral network to predict the $$ of home
	trainX = np.hstack([trainCategorical, trainContinuous])
	testX = np.hstack([testCategorical, testContinuous])

	# return the concatenated training and testing data
	return (trainX, testX)
