import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def weight_update(w_prev, x, y, alpha):
    # x dot w
    x_w_dot = np.dot(x,w_prev)
    # w - alpha ( sig - y ) * deriv_sig * xT
    result = w_prev - alpha* (sigmoid(x_w_dot) - y)*deriv_sigmoid(x_w_dot)*np.reshape(x,(2,1))
    return result

def square_loss(w,test_datas,test_labels):
    #sig(x dot w) - y
    result = sigmoid(np.dot(test_datas,w))-test_labels
    result = np.square(result)
    result = np.sum(result,axis=0)
    return result

if __name__ == '__main__':
    # training datas
    input_datas = np.array([
                    [-1.28, 0.09],
                    [ 0.17, 0.39],
                    [ 1.36, 0.46],
                    [-0.51,-0.32]
                    ])
    print()
    # training labels
    input_labels = np.array([0,1,1,0]).reshape(4,1)
    # parameters
    w0 = np.array([-1, 1]).reshape(2,1)
    alpha = 1
    #testing data
    test_datas = np.array([
                        [-0.5,-1],
                        [0.75,0.25]
                         ])
    test_labels = np.array([0,1]).reshape(2,1)

    w_prev = w0
    #every time trainig with one data from training set
    for index in range(4):
        w_next = weight_update(w_prev, input_datas[index], input_labels[index], alpha)
        #print("w value : ", w_next)
        w_prev = w_next

    result = square_loss(w0, test_datas, test_labels)
    print("square_loss before training : ", result)
    result = square_loss(w_prev, test_datas, test_labels)
    print("square_loss after training : ",result)









    
