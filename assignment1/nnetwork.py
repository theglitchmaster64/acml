import numpy as np
import random
import math

class Layer:
	def __init__(self,nodes,features,weight=None,bias=None):
		self.activity = 0
		if weight == None:
			self.weight = np.random.randn(nodes,features)
		else:
			self.weight = weight
		if bias == None:
			self.bias = np.random.randn(nodes)
		else:
			self.bias = bias

	def activate(self,features):
		res = np.dot(features,self.weight.T) + self.bias
		self.activity = 1.0/(1.0+np.exp(-res))
		return self.activity

class Network:
	def __init__(self):
		self.layers = []

	def add_layer(self,layer):
		self.layers.append(layer)

	def fwd(self,vals):
		for l in self.layers:
			vals = l.activate(vals)
		return vals

	def eval(self,vals):
		return self.fwd(vals)
