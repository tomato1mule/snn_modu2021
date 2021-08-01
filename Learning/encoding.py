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

    return output/255



def encode(image,param):
    timeline = np.arange(0,param.T+param.dt,param.dt)
    n_timesteps = timeline.shape[0]

    freqs = np.interp(image.reshape(-1), [-1.069,2.781], [1/600,20/600])
    assert np.sum(freqs<=0) == 0
    periods = np.ceil(1./freqs)

    
    #random_t0 = np.random.randint(0,n_timesteps//4,freqs.shape)
    #random_t0 = np.random.randint(0,5,freqs.shape)
    random_t0 = 0

    spiketrain = np.zeros((timeline.shape[0],*freqs.shape))
    for i,t in enumerate(timeline):
        if i==0:
            continue
        spiketrain[i] = ((i+random_t0)%np.ceil(periods/param.dt) == 0)*1.

    spiketrain = [spiketrain[:,i] for i in range(spiketrain.shape[1])]
    
    return np.array(spiketrain)