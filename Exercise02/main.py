import numpy as np
import os

#Problem 2.1 Dataset reader
def dataset_reader(folder, data_set="train"):
    #get file path
    file_name = "rt-polarity."+data_set+".vecs"
    file_path = os.path.join(folder,file_name)
    #encoding="utf-8" to solve gbk problem
    with open(file_path, encoding="utf-8") as file:
        lines = file.readlines()
        #Add a bias, i.e. append a trailing 1 to each input vector x
        x_data = np.empty((len(lines),101),dtype=float)
        labels = np.empty((len(lines),1),dtype=int)
        for index, line in enumerate(lines):
            #vector is string type
            _, label, vectors = line.split('\t')
            vectors = vectors.split()
            x_data[index, :100] = [float(f) for f in vectors]
            x_data[index, 100] = 1
            label = label.split('=')[1]
            if label == "POS":
                labels[index] = 1
            else:
                labels[index] = 0
        return x_data, labels

#Problem 2.2 Numpy implementation
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0
    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size,:]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches

def sigmoid(x, derivative=False):
    x = np.clip(x, -500, 500)
    sigm = 1. / (1.+ np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def epoch_update(train_x, train_labels, w0, batch_size, learning_rate):
    w = w0
    train_batches = create_mini_batches(train_x, train_labels, batch_size)

    for X_batch, y_batch in train_batches:
        grad = 0
        for x, y in zip(X_batch, y_batch):
            sigxw = sigmoid(np.dot(x, w))
            grad += (sigxw - y) * sigmoid(x,True) * x
        grad /= len(y_batch)
        w -= learning_rate * grad
    return w

def test(test_x, test_labels, w):
    predictions = [sigmoid(np.dot(x, w)) for x in test_x]
    predictions_discrete = [np.rint(pred) for pred in predictions]

    n = len(test_labels)
    mean_square_loss = sum([(pred - y) ** 2 for pred, y in zip(predictions, test_labels)]) / n
    accuracy = sum([pred == y for pred, y in zip(predictions_discrete, test_labels)]) / n

    return mean_square_loss, accuracy

if __name__ == '__main__':
    #load data set
    train_x, train_labels = dataset_reader("DATA")
    dev_x, dev_labels = dataset_reader("DATA","dev")
    test_x, test_labels = dataset_reader("DATA", "test")

    np.random.seed(seed=5)
    w0 = np.random.normal(0,1,(101))

    for i in range(100):
        w = epoch_update(train_x, train_labels, w0, 10, 0.01)

    loss, accuracy = test(dev_x, dev_labels, w)
    print("Loss on dev after {} epochs: {}, accuracy: {}".format(100, loss, accuracy))








