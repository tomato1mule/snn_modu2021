import numpy as np
import imageio

def learned_weights():
    image_names = [f"{i}" for i in range(6)]
    ans =[]
    for image_name in image_names:
        img = imageio.imread("data/"+ image_name + ".png")
        mask = img.reshape(-1) > 0
        ans.append(
            (1-mask)*-0.7 + mask
        )
    return ans
