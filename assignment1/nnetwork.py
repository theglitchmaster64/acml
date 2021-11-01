import numpy as np
np.random.seed(0)




class Layer:
	def __init__(self,n_nodes,n_in):
		self.weight = np.random.randn(n_nodes,n_in)
		self.bias = np.zeros((1,n_nodes))
		self.output = self.weight.T

	def fwd_prop(self,input): #for each node in input (input*wt)+bias
		self.output = np.dot(input,self.weight.T) + self.bias

	def __repr__(self):
		return 'wt:\n{}\nbias:\n{}'.format(self.weight,self.bias)
