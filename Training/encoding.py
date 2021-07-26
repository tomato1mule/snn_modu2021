import numpy as np
from itertools import product
import torch
import torch.nn as nn
import math


def create_rf_kernel(): #Receptive Field
    kernel = np.zeros((5,5)) # initialize kernel

    cr = 2
    cc = 2
    for i,j in product(range(5),range(5)):
        d = abs(cr-i) + abs(cc-j) # Manhattan distance
        kernel[i,j] = -0.375 * d + 1

    return kernel

# Receptive Field Kernel
sca1 =  0.625
sca2 =  0.125
sca3 = -0.125
sca4 = -.5

RF_kernel = [[	sca4 ,sca3 , sca2 ,sca3 ,sca4],
            [	sca3 ,sca2 , sca1 ,sca2 ,sca3],
            [ 	sca2 ,sca1 , 	1 ,sca1 ,sca2],
            [	sca3 ,sca2 , sca1 ,sca2 ,sca3],
            [	sca4 ,sca3 , sca2 ,sca3 ,sca4]]
RF_kernel = np.array(RF_kernel)


def RF_convolution(image,kernel):
    with torch.no_grad():
        conv = nn.Conv2d(1,1,(5,5),padding=2,bias=False,padding_mode='zeros')
        conv.weight = nn.Parameter(torch.tensor(kernel.reshape(1,1,*kernel.shape),dtype=torch.float32))
        input_tensor = torch.tensor(image.reshape(1,1,*image.shape),dtype=torch.float32)
        output = conv(input_tensor).numpy().reshape(*image.shape)

    return output


def encode_original(image,param):

	#initializing spike train
    train = []
    #periods = []

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
		
            temp = np.zeros([(param.T+1),])

			#calculating firing rate proportional to the membrane potential
            freq = np.interp(image[x][y], [-1.069,2.781], [1,20])


				
            period = math.ceil(600/freq)
            #periods.append(period)

			#generating spikes according to the firing rate
            k = period
            if(image[x][y]>0):
                while k<(param.T+1):
                    temp[k] = 1
                    k = k + period
            train.append(temp)
    return train

def encode(image,param):
    timeline = np.arange(0,param.T+param.dt,param.dt)

    freqs = np.interp(image.reshape(-1), [-1.069,2.781], [1,20])
    periods = np.ceil(600/freqs)

    spiketrain = np.zeros((timeline.shape[0],*freqs.shape))
    for i,t in enumerate(timeline):
        spiketrain[i] = ((i+1)%periods == 0)*1.

    spiketrain = [spiketrain[:,i] for i in range(spiketrain.shape[1])]
    
    return np.array(spiketrain)