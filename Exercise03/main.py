import numpy as np
import os
import tensorflow as tf

batch_size = 5
learning_rate = 0.01
epochs = 200
layer_dim = 20

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


def mini_batches(train_datas, train_labels, batch_size):
  perm = np.random.permutation(len(train_labels))
  datas = train_datas[perm]
  labels = train_labels[perm]
  data_batches = np.array_split(datas, len(train_labels)//batch_size)
  label_batches = np.array_split(labels, len(train_labels)//batch_size)
  return data_batches, label_batches

def add_layer(inputs, in_size, out_size, activation_function=None):
  #weights initialized by the standard normal distribution (mean 0, sigma 1)
  Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0, stddev=1))
  Wx_plus_b = tf.matmul(inputs, Weights)
  if activation_function is None:
     outputs = Wx_plus_b
  else:
     outputs = activation_function(Wx_plus_b)
  return Weights, outputs
  


if __name__ == '__main__':
    train_datas, train_labels = dataset_reader("DATA");
    dev_datas, dev_labels = dataset_reader("DATA", "dev")
    test_datas, test_labels = dataset_reader("DATA", "test")

    print("start")
 
    tf.set_random_seed(9)

    xs = tf.placeholder(tf.float32, shape =(None, 101))
    ys = tf.placeholder(tf.float32, shape=(None, 1))
    #add layers
    Weights1, l1 = add_layer(xs, 101, layer_dim, activation_function=tf.nn.sigmoid)
    Weights2, l2 = add_layer(l1, layer_dim ,layer_dim, activation_function=tf.nn.sigmoid)
    Weights3, l3 = add_layer(l1, layer_dim ,layer_dim, activation_function=tf.nn.sigmoid)
    #output layer
    _, prediction = add_layer(l2, layer_dim, 1,activation_function=tf.nn.sigmoid)

    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    loss =  tf.losses.mean_squared_error(ys,prediction)
    #regularization = tf.nn.l2_loss(Weights1)+tf.nn.l2_loss(Weights2)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction) + 0.01 * regularization) 

    correct_prediction = tf.equal(tf.round(prediction),ys)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Optimizer
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    for i in range(epochs):
      #data_batches, label_batches = get_random_batches(train_datas, train_labels, batch_size)
      data_batches, label_batches = mini_batches(train_datas, train_labels, batch_size)
      for X_batch, y_batch in zip(data_batches, label_batches):
         sess.run(train_step, feed_dict={xs:X_batch, ys:y_batch})
    loss_val = sess.run(loss, feed_dict={xs:dev_datas, ys:dev_labels})
    acc_val = sess.run(accuracy, feed_dict={xs:dev_datas, ys:dev_labels})
    print("loss_val with dev dataset: ", loss_val)
    print("acc_val with dev dataset: ", acc_val)
    loss_val = sess.run(loss, feed_dict={xs:test_datas, ys:test_labels})
    acc_val = sess.run(accuracy, feed_dict={xs:test_datas, ys:test_labels})
    print("||loss_val with test dataset: ", loss_val)
    print("||acc_val with test dataset: ", acc_val)
    print("end###")
