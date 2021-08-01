import numpy as np


def stdp(t,tau_pre,tau_post,A_pre,A_post):
    if t>0 :
        return -A_post * np.exp(-float(t)/tau_post)
    elif t<=0:
        return A_pre * np.exp(float(t)/tau_pre)


        

                
            
        

