import numpy as np
import torch
import torch.nn as nn

w = np.array(
    [[-0.5,-0.125,  0.25,  -0.125, -0.5  ],
    [-0.125 , 0.25  , 0.625 , 0.25 , -0.125],
    [ 0.25   ,0.625 , 1. ,    0.625 , 0.25 ],
    [-0.125 , 0.25  , 0.625 , 0.25,  -0.125],
    [-0.5  , -0.125 , 0.25 , -0.125 ,-0.5  ]]
    )
    
with torch.no_grad():
    conv = nn.Conv2d(1,1,(5,5),padding=2,bias=False,padding_mode='zeros')
    conv.weight = nn.Parameter(torch.tensor(w.reshape(1,1,*w.shape),dtype=torch.float32))

def rf(input_):
    P = np.zeros([28,28])
    with torch.no_grad():
        input_tensor = torch.tensor(input_.reshape(1,1,*input_.shape),dtype=torch.float32)
        P = conv(input_tensor).numpy().reshape(*input_.shape)

    
    return P
