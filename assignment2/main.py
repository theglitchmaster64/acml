#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from classes import *
from matplotlib import pyplot as plt


if __name__=='__main__':

	print('ok')
	print('loading samples...')
	(x_train, _), (x_test, _) = keras.datasets.cifar100.load_data()
	x_train_grey = np.zeros((50000,32,32,1))
	x_test_grey = np.zeros((10000,32,32,1))
	for i in range(len(x_train_grey)):
		x_train_grey[i] = np.dot(x_train[i][...,:3],[0.299,0.587,0.144]).reshape((32,32,1))
	for i in range(len(x_test_grey)):
		x_test_grey[i] = np.dot(x_test[i][...,:3],[0.299,0.587,0.144]).reshape((32,32,1))

	x_train = x_train / 255.0
	x_test = x_test / 255.0
	x_train_grey = x_train_grey / 255.0
	x_test_grey = x_test_grey / 255.0

	#encoder stub

	encoder_input = keras.layers.Input(shape=(32,32,1))
	enc_lyr = encoder_input
	enc_lyr = keras.layers.Conv2D(filters=64,kernel_size=3,strides=2,activation='relu',padding='same')(enc_lyr)
	enc_lyr = keras.layers.Conv2D(filters=128,kernel_size=3,strides=2,activation='relu',padding='same')(enc_lyr)
	encoder_shape = keras.backend.int_shape(enc_lyr)
	enc_lyr = keras.layers.Flatten()(enc_lyr)
	encoder_output = keras.layers.Dense(256)(enc_lyr)

	encoder_stub = keras.models.Model(encoder_input,encoder_output,name='mr.enc')
	print(encoder_stub.summary())


	#decoder stub
	decoder_input = keras.layers.Input(shape=256)
	dec_lyr = keras.layers.Dense(encoder_shape[1]*encoder_shape[2]*encoder_shape[3])(decoder_input)
	dec_lyr = keras.layers.Reshape((encoder_shape[1],encoder_shape[2],encoder_shape[3]))(dec_lyr)
	dec_lyr = keras.layers.Conv2DTranspose(filters=128,kernel_size=3,strides=2,activation='relu',padding='same')(dec_lyr)
	dec_output = keras.layers.Conv2DTranspose(filters=3,kernel_size=3,strides=2,activation='sigmoid',padding='same')(dec_lyr)
	dec_lyr = keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=2,activation='relu',padding='same')(dec_lyr)
	dec_ouput = keras.layers.Conv2DTranspose(filters=3,kernel_size=3,strides=2,activation='sigmoid',padding='same')(dec_lyr)

	decoder_stub = keras.models.Model(decoder_input,dec_output,name='mr.dec')
	print(decoder_stub.summary())

	#glue

	print('creating model...')

	ae = keras.models.Model(encoder_input,decoder_stub(encoder_stub(encoder_input)),name='mr.AE')
	print(ae.summary())

	print('compiling model...')

	ae.compile(optimizer='adam',loss='mse')

	epochs=32

	print('training for epochs:{}'.format(epochs))

	history = ae.fit(x_train_grey,x_train,validation_data=(x_test_grey,x_test),epochs=epochs,batch_size=4)
	#plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])
	#plt.title('model loss')
	#plt.xlabel('epoch')
	#plt.ylabel('loss')
	#plt.legend(['train','test'],loc='upper left')

	print('saving chart to chart_{}.png'.format(epochs))
	#plt.savefig('chart_{}.png'.format(epochs))
	print('chart saved to chart_{}.png'.format(epochs))

	print('predicting 10000 colorized images from TEST_GREY')

	y = ae.predict(x_test_grey)
	Y = OutputImages(input_images=x_test_grey,output_images=y,name='finaltest')
	Y.write_images()

	print('all done!')

	#y2 = ae.predict(x_train_grey)
	#Y2 = OutputImages(input_images=x_train_grey,output_images=y2)
	#Y2.write_images()
