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
def sigmoid(x):
    x = np.clip(x, -500, 300)
    x = 1. / (1.+ np.exp(-x))
    return x

def random_mini_batches(train_x, train_labels, batch_size):
    length = len(train_x)
    perm = np.random.permutation(length)
    X = train_x[perm]
    y = train_labels[perm]
    X_batches = np.array_split(X, length//batch_size)
    y_batches = np.array_split(y, length//batch_size)
    return zip(X_batches, y_batches)

def epoch_calc(train_x, train_labels, w, batch_size, learning_rate):
    mini_batches = random_mini_batches(train_x, train_labels, batch_size)
    for X_batch, y_batch in mini_batches:
        grad = 0
        for X, y in zip(X_batch, y_batch):
            sig = sigmoid(np.dot(X, w))
            grad += (sig - y) * sig * (1-sig) * X
        # average the gradient
        grad /= len(y_batch)
        w -= learning_rate * grad
    return w

def loss_accuracy_func(test_x, test_labels, w):
    preds = [sigmoid(np.dot(x, w)) for x in test_x]
    pred_values = [np.rint(pred) for pred in preds]
    n = len(test_labels)
    square_loss = sum([(pred - y) ** 2 for pred, y in zip(preds, test_labels)]) / n
    accuracy = sum([pred == y for pred, y in zip(pred_values, test_labels)]) / n
    return square_loss, accuracy

if __name__ == '__main__':
    #load data set
    train_x, train_labels = dataset_reader("DATA")
    dev_x, dev_labels = dataset_reader("DATA","dev")
    test_x, test_labels = dataset_reader("DATA","test")

    np.random.seed(seed=9)
    w = np.random.normal(0, 1, (101))
    #epochs = 100
    print("############################ Epoch 100 ####################################")
    w100 = w
    for i in range(100):
        w100 = epoch_calc(train_x, train_labels, w100, 10, 0.01)
    loss, accuracy = loss_accuracy_func(dev_x, dev_labels, w100)
    print("dev dataset loss : ",loss," and accuray : ", accuracy)
    loss, accuracy = loss_accuracy_func(test_x, test_labels, w100)
    print("test dataset loss : ", loss, " and accuray : ", accuracy)

    # epochs = 500
    print("############################ Epoch 500 ####################################")
    w500 = w
    for i in range(500):
        w500 = epoch_calc(train_x, train_labels, w500, 10, 0.01)
    loss, accuracy = loss_accuracy_func(dev_x, dev_labels, w500)
    print("dev dataset loss : ", loss, " and accuray : ", accuracy)
    loss, accuracy = loss_accuracy_func(test_x, test_labels, w500)
    print("test dataset loss : ", loss, " and accuray : ", accuracy)
