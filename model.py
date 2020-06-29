# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 20:57:40 2018

@author: anjan
"""

#Data preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
digits = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
X = digits.iloc[:,1:].values
y = digits.iloc[:,0].values
#normalize
X = X/255.0
test = test/255.0

#reshape
X = X.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#label encoding
y = to_categorical(y, num_classes=10)

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=2)

#build the model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten
classifier = Sequential()
classifier.add(Convolution2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(10, activation='softmax'))

#compiling CNN
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test))

#predict values from validation set
Y_pred = classifier.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1)
#plot confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(Y_true, Y_pred_classes)

#predicting test set
results = classifier.predict(test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name="label")

g = plt.imshow(test[2][:,:,0])

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist_submission.csv",index=False)