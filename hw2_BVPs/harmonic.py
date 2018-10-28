#set up function
def harmonic(y, t, param):
    k, En = param

    y1 = y[1]
    y2 = (k-En)*y[0]
    return [y1, y2]
