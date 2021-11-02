#!/usr/bin/env python3

import sys
import numpy as np
import random
from nnetwork import *

if __name__=='__main__':
	_tmp = []
	dataset = []
	for i in range(0,8):
		_tmp.append('0'*abs(8-i)+'1'+'0'*i)
	for data in _tmp:
		dataset.append([int(x) for x in data])
	del _tmp
	print(dataset)

	n = Network()
	l1 = Layer(8,1)
	
