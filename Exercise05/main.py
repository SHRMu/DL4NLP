# import psutil
# import os
# import gc
import numpy as np
from numpy import zeros
from numpy import array
from numpy.random import choice

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.models import model_from_json
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences

embedding_dim = 50
max_len = 7 # N_gram
# poem_demo = ["<sos/>","<sos/>","<sos/>","<sos/>","<sos/>","<sos/>","<sos/>"]
# batch_size = 512
batch_size = 128
# vocab_size = 4334
poem_start = 0 # start index
poems_num = 100 #trainig set number
# valid_poems_num = 500
epochs_num = 30 #training_epochs number
new_poems_num = 10

peoms_path = "./train_poems.txt" # train_file
embeddings_path = "./glove.6B.50d.txt"

#Problem 2.2 Data Input and Preprocessing
#read poems from train_file and add some tokens as requested
def file_reader(path):
  file = open(path, encoding='utf-8', mode = "r")
  lines = file.readlines()
  file.close()
  poems = []
  poem = ["<sos/>","<sos/>","<sos/>","<sos/>","<sos/>","<sos/>","<sos/>"]
  for line in lines:
    if line == '\n':
      poem = poem[:-1]
      poem.append("<eos/>") # end of a poem
      poems.append(poem)
      poem = ["<sos/>","<sos/>","<sos/>","<sos/>","<sos/>","<sos/>","<sos/>"] # add start of sentence
    else:
      line = line.split('\n')[0] #remove '\n' at the end of each line
      tokens = line.split(" ")
      tokens.append("<eol/>")
      for token in tokens:
        poem.append(token)
  return poems

# calculate the vocab size in train poems
def poems_processing(poems):
  # build new dict to count words in poems
  allWords = {'<oov/>': 10} 
  for poem in poems:
    for word in poem:
      if word not in allWords:
        #print(word)
        allWords[word] = 1
      else:
        allWords[word] = allWords[word]+ 1
  # delete words which less than 2 times used in poems
  erase = []
  for key in allWords:
    if allWords[key] < 3:
        erase.append(key)
  #print(erase)
  for key in erase:
    allWords['<oov/>'] = allWords['<oov/>']+ allWords[key]
    del allWords[key]
  allWords['<oov/>'] = allWords['<oov/>']-10
  # sort allWords by counting times: key:word, value:count
  wordPairs = sorted(allWords.items(), key = lambda x: -x[1])
#   print(wordPairs)
  words, a= zip(*wordPairs)
  # word to ID, most frequently used word is assigned with a short id
  word2int = dict(zip(words, range(len(words)))) 
#   print(word2int)
  # ID to word
  int2word = dict(zip(word2int.values(), word2int.keys()))
  #print(reverse_dictionary)
  return allWords, word2int, int2word

# N_gram_processing for only one poem
def N_gram_processing(poem, n):
  N_gram_data=[]
  N_gram_label=[]
  for i in range(len(poem)-n):
    N_gram = poem[i:i+n+1]
    N_gram_data.append(tuple(N_gram[:-1]))
    N_gram_label.append(N_gram[-1])
  return N_gram_data, N_gram_label
  
# Problem 2.3 Embedding Preparation
# load pretrained embedding from glove.6B.50d.txt
# return pre_embeddings as word2Vec
def load_glove_embeddings(file_path):
  with open(file_path) as file:
    #words_list = []
    pre_embeddings = {}
    for line in file:
      l = line.split(" ")
      word_vetor = [float(w) for w in l[1:]]
      pre_embeddings [l[0]] = word_vetor
      #words_list.append(l)
  pre_embeddings ["<sos/>"] = [float(w) for w in np.random.rand(1, 50)[0]]
  pre_embeddings ["<eol/>"] = [float(w) for w in np.random.rand(1, 50)[0]]
  pre_embeddings ["<eos/>"] = [float(w) for w in np.random.rand(1, 50)[0]]
  pre_embeddings ["<oov/>"] = [float(w) for w in np.random.rand(1, 50)[0]]
  return pre_embeddings 

# return embeddings as int2Vec
def embedding_processing(dictionary, pre_embeddings):
  embeddings = {}
  words = dictionary.keys()
  for count, word in enumerate(words):
    #print(count," ",pre_embeddings[word])
    #Todo: how to deal with uppercase vocal like "and" "And"
    if word in pre_embeddings.keys():
      embeddings[count] = pre_embeddings[word]
    elif word.lower() in pre_embeddings.keys():
      embeddings[count] = pre_embeddings[word.lower()]
    else:
      embeddings[count] = pre_embeddings["<oov/>"]
  return embeddings

# replace word in poems with integer
def poems_encoding(poems, word2int):
  poems_integer = []
  for poem in poems:
    poem_integer = []
    for word in poem:
      if word in word2int.keys():
        poem_integer.append(word2int[word])
      else:
        poem_integer.append(word2int["<oov/>"])
    poems_integer.append(poem_integer)
  return poems_integer

def load_data_set(poems, max_len):
  poems = poems_encoding(poems, word2int)
  datas = []
  labels = []
  for poem in poems:
    data, label=N_gram_processing(poem, max_len)
    datas = datas+list(data)
    labels = labels+label
#   print(len(datas))
#   print(len(labels))
  X_data = np.empty((len(labels), max_len), dtype = int)
  y_label = np.empty((len(labels),vocab_size), dtype = int)
  for index, _ in enumerate(labels):
    X_data[index, : ] = [int(x) for x in list(datas[index])]
    y_label[index] =  to_categorical(labels[index], vocab_size)
  del datas, labels
  return X_data, y_label

def batch_generator(data, targets, batch_size):
  batches = (len(data) + batch_size - 1)//batch_size
  while(True):
     for i in range(batches):
        X = data[i*batch_size : (i+1)*batch_size]
        Y = targets[i*batch_size : (i+1)*batch_size]
        yield (X, Y)

def get_embedding_matrix(dictionary, embeddings):
  embedding_matrix = zeros((vocab_size, embedding_dim), dtype=float)
  for i, key in enumerate(dictionary.keys()):
    embedding_vector = embeddings[dictionary[key]]
    if embedding_vector is not None: 
      embedding_matrix[i] = embedding_vector
    else:
      embedding_matrix[i] = embeddings[dictionary["<oov/>"]]
  return embedding_matrix

def scheduler(epoch):
    # every 10 step reduce lr to 0.1*lr
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

def load_model():
  # load json and create model
  json_file = open('drive/My Drive/DATA/model.json', 'r+')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("drive/My Drive/DATA/model.h5")
  print("Loaded model from disk")
  return loaded_model

def poem_generator(word2int, int2word):
  model = load_model()
#   start_word = "I am here , always waiting for"
  new_poems = []
#   input_index = [word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"]]
  input_index = [word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"]]
#   for word in start_word.split(" "):
#     input_index.append(word2int[word])
  print("generate poem ... ... ")
  next_index = input_index[len(input_index)-1]
#   poems_count = 0
  while len(new_poems) < new_poems_num:
#   while next_index not in [3]:
    predict_index = model.predict(np.array([input_index[-1*max_len:]]))
#     predict_list = predict_index[0].tolist()
#     print(predict_index[0])
#     print(len(predict_index[0]))
#     while next_index == word2int["<oov/>"]:
    next_index = choice(np.arange(0, vocab_size), p=predict_index[0])
    while next_index == word2int["<oov/>"]:
      next_index = choice(np.arange(0, vocab_size), p=predict_index[0])
#     print(next_index)
    if next_index == word2int["<eos/>"]:
      new_poems.append(input_index)
      input_index = [word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"],word2int["<sos/>"]]
    #print(next_index)
    else: 
      input_index.append(next_index)
  print("### End generator ... ... ")
  #print(input_index)
  output_words = []
  for poem in new_poems:
    output_word = []
    for index in poem:  
      word = int2word[index]
      if word == "<sos/>":
        output_word.append("")
      elif word == "<eol/>":
        output_word.append("\n")
      elif word == "<eos/>":
        output_word.append("\n")
      else:
        output_word.append(word)
    print(output_word)
    output_words.append(output_word[3:])
#   print(output_word)
  new_poems = output_words
  del output_word, output_words
#   return output_word
  return new_poems
  
def write_poem_to_file(new_poems):
  with open('drive/My Drive/DATA/poems.txt', 'w') as f:
    for poem in new_poems:
      for word in poem:
#         print(word)
        f.write("%s " % word)
      f.write("\n\n")
  print("already write poems to text file locally !!! ")

if __name__ == '__main__':
    poems = file_reader(peoms_path)
    # print(len(poems)) # total num of poems: 20219
    # print(poems[0])
    # valid_poems = poems[-1*valid_poems_num:] # validation data
    poems = poems[poem_start:poem_start + poems_num] # training data
    print("### Problems2.2a : tokenized poem_1 : ")
    # print(len(poems[0]))
    print(poems[0])

    #first training N_gram testing
    N_gram_data, N_gram_label = N_gram_processing(poems[0], max_len)
    print("### Problem2.2b : N_gram preprocessed poem_1 : ")
    print("N-gram tuple list : ", N_gram_data)
    print("label list : ", N_gram_label)

    allWords, word2int, int2word = poems_processing(poems)

    pre_embeddings = load_glove_embeddings(embeddings_path)
    embeddings = embedding_processing(word2int, pre_embeddings)
    del pre_embeddings

    #print(pre_embeddings["and"])
    #print(len(pre_embeddings["and"]))
    #print(embeddings[0])
    #print(len(embeddings[2]))

    # print("Vocal Size in all poems : ", len(allWords)) # 6362
    vocab_size = len(allWords)

    # new_poems = poem_generator(word2int, int2word)
    # write_poem_to_file(new_poems)

    print("loading data ... ...")

    train_data, train_label = load_data_set(poems, max_len)
    # valid_data, valid_label = load_data_set(valid_poems, max_len)
    print("size of training data : ", len(train_data))
    print("### Problem2.3 : encoded poem 1 according to the index :")
    print(train_data[:len(N_gram_label)])
    print(train_label[:len(N_gram_label)])
    del poems

    embedding_matrix = get_embedding_matrix(word2int, embeddings)
    del embeddings, allWords, word2int, int2word
    #print(embedding_matrix[2])

    print("finished data preparation !!! ")

    print("### start model building ")

    model = Sequential()
    # Embedding Layer
    model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
    # Hidden Layer
    # model.add(LSTM(3500, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(512))
    # Output Layer
    model.add(Dense(vocab_size, activation='softmax'))
    # compile the model
    adam = Adam(lr=0.01)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())

    # fit the model
    # reduce_lr = LearningRateScheduler(scheduler)
    model.fit(train_data, train_label, epochs=epochs_num)
    # model.fit_generator(generator = batch_generator(train_data, train_label, batch_size),
    #                     steps_per_epoch = (len(train_data)+batch_size-1)//batch_size,
    #                     epochs = epochs_num,
    # #                     callbacks=[reduce_lr]
    # )

    # loss, accuracy = model.evaluate(valid_data, valid_label, batch_size=valid_poems_num)
    # print('Accuracy of Valid_data: %f' % (accuracy*100))

    # serialize model to JSON
    # model_json = model.to_json()
    # with open("drive/My Drive/DATA/model.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("drive/My Drive/DATA/model.h5")
    # print("Saved model to disk")

