import numpy as np

# define function and update rule here
# sigmoid function
def sig(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid
def dsig(x):
    return np.multiply(sig(x), (x / x - sig(x)))


# update of weights
def update(w_old, x, y, alpha):
    y_pred = np.matmul(x, w_old.T)
    w_new = w_old - alpha * (sig(y_pred) - y) * dsig(y_pred) * x
    return w_new

# Loss
def square_loss(y_true, y_pred):
    loss = np.multiply(y_pred - y_true, y_pred - y_true)
    return np.sum(loss)

if __name__ == '__main__':
    w0 = np.matrix('-1,1')
    w_old = w0
    x_train = np.matrix('-1.28,0.09;0.17,0.39;1.36,0.46;-0.51,-0.32')
    y_train = np.matrix('0;1;1;0')
    x_test = np.matrix('-0.50,-1.00;0.75,0.25')
    y_test = np.matrix('0;1')
    alpha = 1
    N = 1
    # Training
    # for i in range(4):   # uncomment it if you want to train it for more times
    for idx in range(4):
        w_new = update(w_old, x_train[idx], y_train[idx], alpha)
        w_old = w_new
        print(w_old)
    # Testing
    # before training
    y_before = sig(np.matmul(x_test, w0.T))
    square_loss(y_before, y_test)

    # after training
    y_after = sig(np.matmul(x_test, w_new.T))
    square_loss(y_after, y_test)
