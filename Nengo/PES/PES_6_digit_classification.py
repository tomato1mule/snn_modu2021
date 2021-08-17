import numpy as np
import cv2
import matplotlib.pyplot as plt

import nengo

T = 1.
max_digit = 6
dim = 784

def digit_choice(t):
    return (int(t//T) % max_digit)

def input_signal(t):
    digit = digit_choice(t)
    image = cv2.imread(f"data/{digit}.png",0).reshape(-1)
    return image

def target_signal(t):
    onehot = np.zeros(max_digit)
    digit = digit_choice(t)
    onehot[digit] = 1
    return onehot


model = nengo.Network()
with model:
    image_node = nengo.Node(output=input_signal)
    label_node = nengo.Node(output=target_signal)
    pre = nengo.Ensemble(n_neurons=50, dimensions=dim)
    post = nengo.Ensemble(n_neurons=50, dimensions=max_digit)
    target = nengo.Ensemble(n_neurons=50, dimensions=max_digit)
    error = nengo.Ensemble(n_neurons=50, dimensions=max_digit)
    
    connection_to_learn = \
        nengo.Connection(pre,post,function=lambda _:np.random.random(max_digit))
    connection_to_learn.learning_rule_type = nengo.PES(learning_rate=3e-4)
    nengo.Connection(error,connection_to_learn.learning_rule)    
    
    
    nengo.Connection(label_node,target)
    
    
    nengo.Connection(image_node, pre)
    nengo.Connection(post, error)
    nengo.Connection(target,error,transform=-1)
    
    stop_learning = nengo.Node(output=lambda t: t >= 24)
    nengo.Connection(
        stop_learning, error.neurons, transform=-20 * np.ones((error.n_neurons, 1))
    )

