import numpy as np
from neuron import *

class Synapse:
    def __init__(self,input_size, postsynapse, weight=None):
        self.input_size = input_size # of Presynaptic Neurons
        self.post = postsynapse
        if weight:
            self.weight = weight # (Pre, Post)
        else:
            self.random_initialize(0,1)
        assert self.weight.shape == (self.input_size, len(self.post))

    def run(self,input_spikes, t, adaptation = 1):
        assert input_spikes.shape[0] == self.input_size
        epsp = input_spikes @ self.weight * adaptation

        output_spikes = np.zeros(len(self.post))
        for i,neuron in enumerate(self.post):
            output_spikes[i] = neuron.run(epsp[i],t)

        return output_spikes

    def update_weight(self,weight):
        self.weight = weight

    def random_initialize(self,low,high):
        #self.weight = np.random.randint(low,high,size=(self.input_size, len(self.post)))
        self.weight = np.random.uniform(low,high,size=(self.input_size, len(self.post)))
    

    def identity_initialize(self,mult=1):
        assert(self.input_size == len(self.post))
        self.weight = np.eye(self.input_size) * mult

    def reset(self):
        for neuron in self.post:
            neuron.reset()
    

        

                
            
        

