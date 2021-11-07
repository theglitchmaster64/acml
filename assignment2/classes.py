import cv2
import numpy as np
import string
import random
import os

# load n samples from samples folder and rescale to 32x32
# m splits the data into train and test, ex - m=8 is a 80-20 split
# each sample is a dict containing a rgb and greyscale version of the same image
class LoadSamples:
	def __init__(self,n,m):
		self.samples = []
		for i in range(1,n+1):
			self.samples.append({'col':cv2.resize(cv2.imread('samples/{}.jpg'.format(i)),(32,32))/255.0,'grey':cv2.resize(cv2.imread('samples/{}.jpg'.format(i),cv2.IMREAD_GRAYSCALE),(32,32))/255.0})
		self.train = self.samples[0:m]
		self.test = self.samples[m::]

	def cache_samples(self):
		for i in range(len(self.samples)):
			cv2.imwrite('tmp/bw_{}.jpg'.format(i),self.samples[i]['grey'])
			cv2.imwrite('tmp/{}.jpg'.format(i),self.samples[i]['col'])

class OutputImages:
	def __init__(self,input_images,output_images,name=''.join([chr(random.randrange(97,122)) for x in range(8)])):
		self.input_images = []
		self.output_images = []
		self.name = name
		for i in input_images:
			self.input_images.append(i*255.0)
		for j in output_images:
			self.output_images.append(j*255.0)


	def write_images(self):
		if os.path.exists(os.path.join(os.getcwd(),self.name)):
			print('path already exists! aborting...')
			return None
		else:
			os.mkdir(os.path.join(os.getcwd(),self.name))
		for i in range(len(self.output_images)):
			cv2.imwrite(os.path.join(os.getcwd(),self.name)+'/out_{}.jpg'.format(i),self.output_images[i])
			cv2.imwrite(os.path.join(os.getcwd(),self.name)+'/in_{}.jpg'.format(i),self.input_images[i])
