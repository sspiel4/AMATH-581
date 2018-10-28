#set up function
def nonlinear_harmonic(y, t, param):
    k, En, gamma = param

    y1 = y[1]
    y2 = ((gamma*y[0]**3)+(k-En))*y[0]
    return [y1, y2]
