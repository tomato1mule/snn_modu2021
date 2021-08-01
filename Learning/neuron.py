import numpy as np

class NeuronParam:
    def __init__(self,  t_refractory=30,
                        V_rest = 0,
                        V_thr = 5,
                        V_min = -500,
                        V_spike = None,
                        V_inhibit = -500,
                        leak = 0.7,
                        record = False):
        self.t_refractory = t_refractory
        self.V_rest = V_rest
        self.V_thr = V_thr
        self.V_min = V_min
        self.V_inhibit = V_inhibit
        self.leak = leak

        if V_spike: #Only for Visualization
            self.V_spike = V_spike
        else:
            self.V_spike = V_thr * 2 





class SimpleNeuron:
    def __init__(self,param):
        self.t_refractory = param.t_refractory
        self.t_rest = -1
        self.V = param.V_rest
        self.V_rest = param.V_rest
        self.V_thr = param.V_thr
        self.V_min = param.V_min
        self.V_spike = param.V_spike  #Only for Visualization
        self.V_inhibit = param.V_inhibit
        self.leak = param.leak

        self.V_history = []
        self.spike_history = []
        self.input_history =[]
        
    def run(self,input,t):

        # Rest or Excite
        if not self.t_rest < t: # If Refractory
                self.V = self.V_rest
        else: # If Not Refractory
            self.V += input


        # Recovery to V_rest
        #if self.V > self.V_rest: # If Depolarized
        #    self.V -= self.leak 
        #elif self.V < self.V_rest: # If Hyperpolarized
        #    self.V += self.leak
        self.V -= self.leak 
        # V_min
        if self.V < self.V_min:
            #self.V = self.V_min
            self.V = self.V_rest


        # Record V, Input, Spikes
        self.V_history.append(self.V)
        self.input_history.append(input)
        self.spike_history.append(0)

        return self.V
    
    def is_spikable(self,t):
        if self.V >= self.V_thr and self.t_rest < t:
            return True
        else:
            return False

    def check_spike(self,t):
        # Spike
        if self.is_spikable(t):
            self.V = self.V_rest # Reset V
            self.t_rest = t + self.t_refractory # Go to Refractory Period

            # Change Recording
            self.spike_history[-1] = 1
            self.V_history[-1] = self.V_spike

            return 1
        else:
            return 0

    def inhibit(self):
        self.V = self.V_inhibit

    def reset(self):
        self.t_rest = -1
        self.V = self.V_rest
        self.V_history = []
        self.spike_history = []
        self.input_history =[]

        

            



