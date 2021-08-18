import numpy as np
import cv2
import matplotlib.pyplot as plt

import nengo

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


download_root = './'

train_dataset = MNIST(download_root, train=True, download=False)
test_dataset = MNIST(download_root, train=False, download=False)

data = np.array(train_dataset.data)/255.
label = np.array(train_dataset.targets)


T = 0.5
n_classes = 10
dim = 784
n_train = 1000

def idx_choice(t):
    return int(t//T)

def input_signal(t):
    idx = idx_choice(t)
    image = data[idx].reshape(-1)
    return image

def target_signal(t):
    onehot = np.zeros(n_classes)
    idx = idx_choice(t)
    ans = label[idx]
    onehot[ans] = 1
    return onehot
    
    
model = nengo.Network()
with model:
    image_node = nengo.Node(output=input_signal)
    label_node = nengo.Node(output=target_signal)
    
    pre = nengo.Ensemble(n_neurons=200, dimensions=dim)
    post = nengo.Ensemble(n_neurons=50, dimensions=n_classes)
    target = nengo.Ensemble(n_neurons=50, dimensions=n_classes)
    error = nengo.Ensemble(n_neurons=50, dimensions=n_classes)
    
    
    
    connection_to_learn = \
        nengo.Connection(pre,post,function=lambda _:np.random.random(n_classes))
    connection_to_learn.learning_rule_type = nengo.PES(learning_rate=1e-4)
    nengo.Connection(error,connection_to_learn.learning_rule)    
    

    
    nengo.Connection(image_node, pre)
    nengo.Connection(label_node,target)
    nengo.Connection(post, error)
    nengo.Connection(target,error,transform=-1)

    stop_learning = nengo.Node(output=lambda t: t >= n_train*T)
    nengo.Connection(
        stop_learning, error.neurons, transform=-20 * np.ones((error.n_neurons, 1))
    )


