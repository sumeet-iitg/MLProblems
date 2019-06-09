import numpy as np
from sklearn.datasets import load_iris

def sigmoid(x):
   return 1/(1 + np.exp(x))

def predict(xs, w, thresh = 0.5):
    probas = 1 - sigmoid(np.dot(w.T, xs))
    predictions = probas > thresh # predict value 1 when prob > threshold
    return 1*predictions

def log_likelihood(ys, xs, w):
    '''
    :param ys:
    :param xs:
    :param W:
    :return:
    '''
    n = len(ys)
    exp_term = np.exp(np.dot(w.T, xs))
    ll_terms = -np.log(1 + exp_term) + ys*np.dot(w.T, xs)
    return np.sum(ll_terms)/n

def computeGrad(ys, xs, w):
    n = len(ys)
    exp_term = np.exp(np.dot(w.T, xs))
    prob_y0 = 1/(1 + exp_term)
    dWs = xs*(ys - (1 - prob_y0))
    return np.sum(dWs, axis=1,keepdims=True)/n

def logisticReg(ys, xs, lr=0.001, max_iter = 10000):
    '''
    :param ys: 1xn
    :param xs: dxn
    :return:
    '''
    w = np.zeros((xs.shape[0],1))
    ll = log_likelihood(ys, xs, w)
    dW = computeGrad(ys, xs, w)
    new_w = w + lr * dW
    curr_ll = log_likelihood(ys, xs, new_w)
    iter_ct = 0
    while iter_ct < max_iter:
        if curr_ll >= ll:
            w = new_w
            ll = curr_ll
        dW = computeGrad(ys, xs, new_w)
        print("Grads:{}".format(dW))
        new_w = new_w + lr*dW
        curr_ll = log_likelihood(ys, xs, new_w)
        iter_ct += 1
        print("Iter:{} Best.LL:{} Curr.LL:{}".format(iter_ct, ll, curr_ll))
    return w, ll

if __name__=="__main__":
    iris = load_iris()
    x = iris.data[:, :2]
    y = (iris.target != 0) * 1
    x0 = np.ones(len(y))
    xs = np.vstack([x0.T, x.T])
    w, ll = logisticReg(y.T, xs)
    preds = predict(xs, w)
    n = len(y)
    correct = np.sum(preds == y)
    print("Likelihood:{} Accuracy:{} Weights:{}".format(ll, correct/n, w))