import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def weight_update(w_prev, x, y, alpha):
    x_w_dot = np.dot(x,w_prev)
    result = w_prev - alpha* (sigmoid(x_w_dot) - y)*deriv_sigmoid(x_w_dot)*np.reshape(x,(2,1))
    print(np.reshape(result,(1,2)))
    return result

if __name__ == '__main__':
    # training data
    input_datas = np.array([
                    [-1.28, 0.09],
                    [ 0.17, 0.39],
                    [ 1.36, 0.46],
                    [-0.51,-0.32]
                    ])
    input_labels = np.array([0,1,1,0]).reshape(4,1)
    w0 = np.array([-1, 1]).reshape(2,1)
    alpha = 1
    #testing data
    test_datas = np.array([
                        [-0.5,-1],
                        [0.75,0.25]
                         ])
    test_labels = np.array([0,1]).reshape(2,1)

    w_prev = w0

    for index in range(4):
        w_next = weight_update(w_prev, input_datas[index], input_labels[index], alpha)
        print("w value : ", w_next)
        w_prev = w_next









    
