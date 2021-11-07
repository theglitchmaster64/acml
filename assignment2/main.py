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
	X = LoadSamples(100,80)
	x_train = np.array([x['col'] for x in X.train])
	x_train = x_train.reshape(x_train.shape[0],32,32,3)
	x_test = np.array([x['col'] for x in X.test])
	x_test = x_test.reshape(x_test.shape[0],32,32,3)
	x_train_grey = np.array([x['grey'] for x in X.train])
	x_train_grey = x_train_grey.reshape(x_train_grey.shape[0],32,32,1)
	x_test_grey = np.array([x['grey'] for x in X.test])
	x_test_grey = x_test_grey.reshape(x_test_grey.shape[0],32,32,1)

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

	ae.fit(x_train_grey,x_train,validation_data=(x_test_grey,x_test),epochs=16,batch_size=4)
