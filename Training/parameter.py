import numpy as np
from neuron import NeuronParam

class TrainParam:
    def __init__(self,
                 T=200,
                 dt=1):
        self.T = T
        self.dt = dt