import numpy as np

class NeuronParam:
    def __init__(self,  t_refractory=30,
                        V_rest = 0,
                        V_thr = 5,
                        V_min = -500,
                        V_spike = None,
                        leak = 0.7,
                        record = False):
        self.t_refractory = t_refractory
        self.V_rest = V_rest
        self.V_thr = V_thr
        self.V_min = V_min
        self.leak = leak
        self.record = record
        if V_spike:
            self.V_spike = V_spike
        else:
            self.V_spike = V_thr * 1.5




class SimpleNeuron:
    def __init__(self,param):
        self.t_refractory = param.t_refractory
        self.t_rest = -1
        self.V = param.V_rest
        self.V_rest = param.V_rest
        self.V_thr = param.V_thr
        self.V_min = param.V_min
        self.leak = param.leak
        self.record = param.record
        self.V_spike = param.V_spike
        if self.record:
            self.V_history = []
            self.spike_history = []
            self.epsp_history =[]
        
    def run(self,input,t):
        # Record V
        if self.record:
            self.V_history.append(self.V)



        # Refractory Period
        if not self.t_rest < t:
            # Record No Spike
            if self.record:
                self.spike_history.append(0)
                self.epsp_history.append(0)
            
            return 0
        


        # Not refractory -> EPSP
        self.V += input
        if self.record:
            self.epsp_history.append(input)


        # Spike
        if self.V >= self.V_thr:
            self.V = self.V_rest
            self.t_rest = t + self.t_refractory

            # Record spike
            if self.record:
                self.spike_history.append(1)
                self.V_history[-1] = self.V_spike
            
            
            return 1

        # Recovery to V_rest
        if self.V > self.V_rest: # Depolarized
            self.V -= self.leak 
        elif self.V < self.V_rest: # Hyperpolarized
            self.V += self.leak

        # V_min
        if self.V < self.V_min:
            self.V = self.V_min

        # Record no spike
        if self.record:
            self.spike_history.append(0)

        return 0

    def inhibit(self):
        self.V = self.V_min
    
    def SetThreshold(self, V_thr):
        self.V_thr = V_thr

    def SetLeak(self, leak):
        self.leak = leak

        

            



