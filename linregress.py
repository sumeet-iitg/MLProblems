import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def computeLoss(ys, xs, W):
    data_sz = len(ys)
    sq_err = (np.dot(W.T,xs) - ys)**2
    avg_loss = np.sqrt(np.sum(sq_err))/data_sz
    return avg_loss

def computeGradNorm(ys,xs, W):
    data_sz = len(ys)
    dWs =  (np.dot(W.T,xs) - ys)*xs
    dW_norm = np.sum(dWs, axis=1)/(2*data_sz)
    return dW_norm

def linRegress(ys, xs, lr=0.0001, max_iter = 10000):
    '''
        xs: dim x n 
    '''
    
    W = np.random.uniform(-1,1, size=(xs.shape[0],1))
    loss = computeLoss(ys, xs, W)
    prev_loss = float("inf")
    iter_ct = 0
    while iter_ct < max_iter:
        dW = computeGradNorm(ys, xs, W)
        print "grads:{}".format(dW)
        W = W - lr * dW
        prev_loss = loss
        loss = computeLoss(ys, xs, W)
        iter_ct += 1
        print "Iter:{} Avg Loss:{}, Prev:{}".format(iter_ct, loss, prev_loss)
    
    return W, loss

if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (20.0, 10.0)   
    
    data = pd.read_csv('data/student.csv')
    print(data.shape)
    
    math_val = data['Math'].values
    read_val = data['Reading'].values
    write_val = data['Writing'].values
#
#    # Ploting the scores as scatter plot
#    fig = plt.figure()
#    ax = Axes3D(fig)
#    ax.scatter(math_val, read_val, write_val, color='#ef1234')
#    fig.savefig('student.png')
    m = len(math_val)
    x0 = np.ones(m)
    xs = np.array([x0, math_val, read_val]).T
    W, avg_loss = linRegress(write_val.T, xs.T)
    
    