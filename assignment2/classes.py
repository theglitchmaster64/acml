import cv2
import numpy as np


# load n samples from samples folder and rescale to 32x32
# m splits the data into train and test, ex - m=8 is a 80-20 split
# each sample is a dict containing a rgb and greyscale version of the same image
class LoadSamples:
	def __init__(self,n,m):
		self.samples = []
		for i in range(1,n+1):
			self.samples.append({'col':cv2.resize(cv2.imread('samples/{}.jpg'.format(i)),(32,32)),'grey':cv2.resize(cv2.imread('samples/{}.jpg'.format(i),cv2.IMREAD_GRAYSCALE),(32,32))})
		self.train = self.samples[0:m]
		self.test = self.samples[m::]

	def cache_samples(self):
		for i in range(len(self.samples)):
			cv2.imwrite('tmp/bw_{}.jpg'.format(i),self.samples[i]['grey'])
			cv2.imwrite('tmp/{}.jpg'.format(i),self.samples[i]['col'])
