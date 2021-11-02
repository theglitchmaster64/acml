import numpy
import random
import math

class Node:
	def __init__(self,weight,bias):
		self.weight = weight
		self.bias = bias
		self.state = False
		self.output = []
		self.inputs = []

	def activate(self,input):
		self.inputs = input
		for i in input:
			self.outputs.append(1.0/(1.0+math.exp(-(self.weight*i)+bias)))
		if self.output + bias > 0:
			self.state = True
		else:
			self.state = False
		return self.state

	def __repr__(self):
		return 'wt:{0}\tbias:{1}\n'.format(self.weight,self.bias)


class Layer:
	def __init__(self,type):
		self.type = type
		self.nodes = list()

	def add_node(self,node):
		if type(node) == Node:
			self.nodes.append(node)
			return True
		else:
			return False

	#def prop_fw(self,next_layer):
		#for node in self.nodes:


	def __repr__(self):
		ret_str = self.type + '({} nodes)'.format(len(self.nodes)) + ':\n'
		for i in range(len(self.nodes)):
			ret_str += str(self.nodes[i])+''
		return ret_str


class NNet:
	def __init__(self, n_in, n_hid, n_out):
		self.hidden = Layer(type='hidden')
		for i in range(n_hid):
			self.hidden.add_node(Node(random.random(),random.random()))
		self.output = Layer(type='output')
		for i in range(n_out):
			self.output.add_node(Node(random.random(),random.random()))
		self.network = {'hidden':self.hidden,'output':self.output}
