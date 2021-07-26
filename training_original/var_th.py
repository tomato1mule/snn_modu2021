############################################## README #################################################

# This calculates threshold for an image depending upon its spiking activity.

########################################################################################################


import numpy as np
from neuron import neuron
import random
from matplotlib import pyplot as plt
from recep_field import rf
import cv2
from spike_train import encode
from rl import rl
from rl import update
from reconstruct import reconst_weights
from parameters import param as par
import os


def threshold(train):

	tu = np.shape(train[0])[0]    #Time length of spiketrain
	thresh = 0
	for i in range(tu):
		simul_active = sum(train[:,i])   # total activation level at time i
		if simul_active>thresh:
			thresh = simul_active    # save maximum total activation

	return (thresh/3)*par.scale   # 1/3 of maximum total activation as threshold


