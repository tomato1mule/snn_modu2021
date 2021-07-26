######################################################## README #############################################################

# This file generates rate based spike train from the potential map.

############################################################################################################################


import numpy as np
from numpy import interp
from neuron import neuron
import random
from matplotlib import pyplot as plt
from recep_field import rf
import cv2
from rl import rl
from rl import update
import math
from parameters import param as par

def encode(pot):

	#initializing spike train
	train = []

	for l in range(par.pixel_x):
		for m in range(par.pixel_x):
		
			temp = np.zeros([(par.T+1),])

			#calculating firing rate proportional to the membrane potential
			freq = interp(pot[l][m], [-1.069,2.781], [1,20])
			
			# print freq
			if freq<=0:
				print error
				
			freq1 = math.ceil(600/freq)

			#generating spikes according to the firing rate
			k = freq1
			if(pot[l][m]>0):
				while k<(par.T+1):
					temp[k] = 1
					k = k + freq1
			train.append(temp)
			# print sum(temp)
	return train


