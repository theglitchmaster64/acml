#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from classes import *
from matplotlib import pyplot as plt


if __name__=='__main__':

	print('ok')

	input_shape = (32,32,3)
	red_dim = 256
	layer_filters = [64,128,256]

	#encoder stub
	print('preparing encoder stub...')
	input_layer = keras.layers.Input(shape=input_shape)
	encoder_layer = input_layer
	for nlayer in layer_filters:
		encoder_layer = keras.layers.Conv2D(filters = nlayer, kernel_size = 3,strides=1,activation='relu',padding='same')(encoder_layer)
	shape = keras.backend.int_shape(encoder_layer)
	encoder_layer = keras.layers.Flatten()(encoder_layer)
	encoder_output = keras.layers.Dense(red_dim)(encoder_layer)
	encoder = keras.models.Model(input_layer,encoder_output,name='enc')

	#decoder stub
	print('preparing decoder stub...')
	decoder_inputs = keras.layers.Input(shape=red_dim)
	decoder_layer = keras.layers.Dense(shape[1]*shape[2]*shape[3])(decoder_inputs)
	decoder_layer = keras.layers.Reshape((shape[1],shape[2],shape[3]))(decoder_layer)
	for nlayer in layer_filters[::-1]:
		decoder_layer = keras.layers.Conv2DTranspose(filters=nlayer,kernel_size = 3, strides = 1, activation='relu', padding='same')(decoder_layer)
	decoder_output = keras.layers.Conv2DTranspose(filters=3,kernel_size = 3, strides = 1, activation='sigmoid', padding='same')(decoder_layer)
	decoder = keras.models.Model(decoder_inputs,decoder_output,name='dec')

	ae = keras.models.Model(input_layer,decoder(encoder(input_layer)),name='autoenc')

	print('loading samples...')

	X = LoadSamples(10,8)
	x_train = X.train['col']
	x_test = X.test['col']
	x_train_grey = X.train['grey']
	x_test_grey = X.test['grey']

	print('training model...')

	ae.fit(x_train_grey,x_train,validation_data=(x_test_grey,x_test),epochs=10,batch_size=4)

	print('alldone!')
