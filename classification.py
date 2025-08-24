import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.utils import to_categorical

import matplotlib.pyplot as plt

from keras.datasets import mnist #MNIST contains the data of almost 70k 28*28 pixels images for training purposes.

(x_train , y_train) , (x_test , y_test) = mnist.load_data()
#x_train has 60k images and y_train has it's labels / outputs
#same with x_test and y_test but with 10k images and labels

x_train.shape   #output - (60000 , 28 , 28)

plt.imshow(x_train[0])

num_pixels = x_train.shape[1] * x_train.shape[2]
#flatten the images into one-dimensional vectors, each of size 1 x (28 x 28) = 1 x 784.

x_train = x_train.reshape(x_train.shape[0] , num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
#Reshape changes the shape from (60k,28,28) 2D array to (60k,784) 1D array
#astype changes the values of array into 32bit floats

x_train /= 255
x_test /= 255
# normalize inputs from 0-255 to 0-1

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#for classification we need to divide our target variable into categories.

num_classes = y_test.shape[1]
#gives the total categories of outputs >>10 {0-9}

#print(num_classes) >> 10

#############################################

#BUILDING THE MODEL

#defining a model
def classification_model():
	
	model = Sequential()
	model.add(Input(shape=(num_pixels,)))
	model.add(Dense(num_pixels , activation = 'relu'))
	model.add(Dense(100 , activation = 'relu'))
	model.add(Dense(num_classes , activation = 'softmax'))
	
	model.compile(optimizer='adam',loss='categorical_crossentropy' , metrics = ['accuracy'])
	return model

#creating a model
model = classification_model()

#fitting the model
model.fit(x_train , y_train , validation_data=(x_test , y_test) , epochs=10  , verbose=2)

#evaluate the model
scores = model.evaluate(x_test , y_test , verbose=0)

print('Accuracy - {}% \n Error - {}'.format(scores[1] , 1-scores[1]))
