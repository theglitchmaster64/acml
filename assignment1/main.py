#!/usr/bin/env python3

import sys
from NNet import *
import nnetwork as nn

if __name__=='__main__':
	input_layer = nn.Layer(8,8)
	hidden_layer = nn.Layer(3,8)
	output_layer = nn.Layer(8,3)
	print('in',input_layer)
	hidden_layer.fwd_prop(input_layer.weight)
	output_layer.fwd_prop(hidden_layer.output)
	print('out',output_layer)
	print('\nOUTPUT:\n')
	print(output_layer.output)
