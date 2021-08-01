import numpy as np
from neuron import *

class Synapse:
    def __init__(self,input_size, post_neurons, weight=None):
        self.post_neurons = post_neurons
        self.input_size = input_size 
        self.output_size = (len(self.post_neurons))
        self.weight = np.zeros((self.output_size,self.input_size))

    def is_spikable(self,t):
        is_spikable = np.zeros(self.output_size)
        for i,neuron in enumerate(self.post_neurons):
            is_spikable[i]=int(neuron.is_spikable(t))

        return is_spikable # Binary Vector with size = self.output_size


    def run(self,input_spikes, t, scale = 1,lateral_inhibition = False):
        result={} # Result to be returned

        #Calculate Input Current
        assert input_spikes.shape[0] == self.input_size
        I = self.weight @ input_spikes.reshape(-1,1) * scale


        #Run Neuros to Calculate Membrane Potential
        Vs = np.zeros(self.output_size)
        for i,neuron in enumerate(self.post_neurons):
            Vs[i] = neuron.run(I[i],t)


        #Check whether V crossed Threshold
        is_spikable = self.is_spikable(t)


        #Calculate Output Spikes
        winner = -1
        output_spikes = np.zeros(self.output_size)
        if np.sum(is_spikable) > 0: #If at least 1 postsynaptic neuron spiked:

            
            if lateral_inhibition: # If Lateral Inhibition
                winner = np.argmax(Vs)
                for i,neuron in enumerate(self.post_neurons):
                    if i != winner:
                        neuron.inhibit()
                    elif i == winner:
                        output_spikes[i] = neuron.check_spike(t)
                        assert output_spikes[i] # If not 1, something's wrong
            else: # No Lateral Inhibition
                for i,neuron in enumerate(self.post_neurons):
                    output_spikes[i] = neuron.check_spike(t)


        
        result['output'] = output_spikes
        result['winner'] = winner 

        return result


    def random_initialize(self,low,high):
        self.weight = np.random.uniform(low,high,size=(self.output_size, self.input_size))
    

    def reset_all_neurons(self):
        for neuron in self.post_neurons:
            neuron.reset()


    def get_spike_record(self):
        output_train = []
        for neuron in self.post_neurons:
            output_train.append(neuron.spike_history)

        return np.array(output_train)  # (output_size, Timesteps)

    def get_V_record(self):
        V_record = []
        for neuron in self.post_neurons:
            V_record.append(neuron.V_history)
        
        return np.array(V_record)  # (output_size, Timesteps)

    def get_I_record(self):
        I_record = []
        for neuron in self.post_neurons:
            I_record.append(neuron.I_history)
        
        return np.array(I_record)  # (output_size, Timesteps)

    


    

        

                
            
        

