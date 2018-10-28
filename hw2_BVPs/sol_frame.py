import numpy as np

def sol_frame(start, stop, delta):
    pts = int(1 + ((stop - start) / delta))
    frame = np.linspace(start, stop, pts)
    return frame
