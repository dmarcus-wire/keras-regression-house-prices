# USAGE
# python keras-regression.py --dataset dataset/houses

# import the necessary packages
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split # creating train/test sets
from submodules import datasets # load and preprocess cat and cont. data
from submodules import models # model architecture
import numpy as np
import argparse # cli argument parser
import locale # format nicely formatted price to screen
import os # operating system specific functions

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
# actual path to the csv file
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
# load attributes from disk
df = datasets.load_house_attributes(inputPath)

# construct training and testing 75% : 25%
print("[INFO] constructing train/test split...")
(train, test) = train_test_split(df, test_size=0.25, random_state=42)

# find largest house price in training and use to scale
# house prices from 0 to 1
# results in better training and convergence
maxPrice = train["price"].max()
# ensure all prices are within 0 and 1
trainY = train["price"] / maxPrice
testY = test["price"] / maxPrice

print("[INFO] processing data...")
(trainX, testX) = datasets.process_house_attributes(df, train, test)

# create MLP and compile use Mean Absolute Percentage Error for Loss
# meaining, we want to minimize the absolutely % diff between priceses we
# predict and actual prices
model = models.create_mlp(trainX.shape[1], regress=True)
# use adam for optimizer
opt = Adam(lr=1e-3, decay=1e-3 / 200)
# compile together
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
# 200 epochs
# batch size of 8
print("[INFO] training model...")
model.fit(x=trainX, y=trainY,
	validation_data=(testX, testY),
	epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
# contain prices of homes based on attributes
preds = model.predict(testX)

# compute the diff between predicted and actual $$$
# subtract the actual prices from the predicts
diff = preds.flatten() - testY
# compute % diff
percentDiff = (diff / testY) * 100
# absolute value
absPercentDiff = np.abs(percentDiff)

# compute the mean and std dev of the abs % diff
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# show some stats on the model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:2f}%, std: {:.2f}%".format(mean, std))

# mean absolute error diff of ~23.1%
# std dev of ~ 22.7%
# this means our model will be off by ~22.7%
# in terms of house price prediction....it's not the greatest
# how do you improve:
# limitation of house price attributes, use a larger dataset
# Boston and Ames house datasets have more aatributes