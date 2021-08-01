import numpy as np


def stdp(t,tau_past,tau_future,A_past,A_future):
    if t>0 :
        return -A_future * np.exp(-float(t)/tau_future)
    elif t<=0:
        return A_past * np.exp(float(t)/tau_past)

def stdp_update(w, dw, scale, lr, w_min, w_max):
    if dw < 0:
        return w + lr*dw*(w-abs(w_min))*scale
    elif dw > 0:
        return w + lr*dw*(w_max-w)*scale
		

        

                
            
        

