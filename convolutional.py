from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Input
from keras.utils import to_categorical

from keras.datasets import mnist

(x_train , y_train) , (x_test , y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')
#reshape format - [samples][height][width][number_of_channels]
#here number of channels is 1 because MNIST is greyscale , for RGB it's 3

x_train /= 255
x_test /= 255
#normalise the pixel values from 0-255 to 0-1

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_train.shape[1]
#y_train.shape = (60k , 10) >> 60k samples each with vector of length 10

################ DEFINING A CONVOLUTIONAL MODEL ################

def convolutional_model():
	model = Sequential()
	model.add(Input(shape=(28 , 28 , 1))) #input is a greyscale with 28*28 pixels

#ADDING FIRST SET OF CONVOLUTIONAL AND POOLING LAYER
	model.add(Conv2D(16,(5,5),strides=(1,1),activation='relu')) #convolutional layer with kernel_size = 5*5 and stride = 1*1
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#ADDING SECOND OF CONVOLUTIONAL AND POOLING LAYER
	model.add(Conv2D(8,(2,2),strides=(1,1),activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(100,activation='relu'))
	model.add(Dense(num_classes,activation='softmax'))
	
	model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
	return model

################ BUILDING A CONVOLUTIONAL MODEL #################

model = convolutional_model()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,batch_size=1024,verbose=2)


scores = model.evaluate(x_test,y_test,verbose=0)
print('Accuracy - {} \n Error - {}'.format(scores[1] , 100-scores[1]*100))
