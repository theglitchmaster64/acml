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
	def __init__(self,name=None,elems=None):
		if elems == None:
			self.items = []
		else:
			self.items = elems
		if name == None:
			self.name = ''.join([chr(random.randrange(97,122)) for x in range(8)])
		else:
			self.name = name

	def add_image(self,img):
		self.items.append(img)

	def write_images(self):
		if os.path.exists(os.path.join(os.getcwd(),self.name)):
			print('path already exists! aborting...')
			return None
		else:
			os.mkdir(os.path.join(os.getcwd(),self.name))
