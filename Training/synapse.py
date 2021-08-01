import numpy as np
from neuron import *

class Synapse:
    def __init__(self,input_size, postsynapse, weight=None):
        self.input_size = input_size # of Presynaptic Neurons
        self.output_size = (len(postsynapse))
        self.post = postsynapse
        if weight:
            self.weight = weight # (Pre, Post)
        else:
            self.random_initialize(0,1)
        assert self.weight.shape == (self.input_size, len(self.post))
        self.recent_activities = np.zeros(self.output_size)

    def run(self,input_spikes, t, adaptation = 1,lateral_inhibition = False):
        assert input_spikes.shape[0] == self.input_size
        epsp = input_spikes @ self.weight * adaptation

        output_spikes = np.zeros(len(self.post))
        for i,neuron in enumerate(self.post):
            output_spikes[i] = neuron.run(epsp[i],t)

        for i,neuron in enumerate(self.post):
            self.recent_activities[i] = neuron.recent_activity

        if lateral_inhibition:
            if np.sum(output_spikes) >= 1:
                spiked_Vs=np.zeros((self.output_size))
                for i,neuron in enumerate(self.post):
                    spiked_Vs[i] = neuron.spiked_V
                winner = np.argmax(spiked_Vs)
                for i,neuron in enumerate(self.post):
                    if i != winner:
                        neuron.inhibit()
                        output_spikes[i] = 0
                print(winner)

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


    

        

                
            
        

