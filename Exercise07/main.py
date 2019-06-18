import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# convert label to one hot vec
def labels_to_onehot(labels):
    labels_onehot = np.zeros((len(labels), 2), dtype=np.int32)
    for i, label in enumerate(labels):
        if label == "POS":
            labels_onehot[i, 0] = 1
        else:
            labels_onehot[i, 1] = 1
    return labels_onehot

# read reviews and labels from file
def file_reader(reviews_file, labels_file):
    reviews = []
    labels = []
    with open(reviews_file, encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            reviews.append(line)
    with open(labels_file, encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            labels.append(line)
    labels = labels_to_onehot(labels)
    return reviews, labels

# load pretrained word2vec file
def word2vec_reader(word2vec_path):
    word2vec_dict = {}
    with open(word2vec_path, encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            words_list = line.split(" ")[:-1]
            if len(words_list) < 300:
                continue
            word2vec_dict[words_list[0]] = words_list[1:]
    word_dim = len(word2vec_dict["</s>"])
    return word2vec_dict, word_dim

def text_word2vec(word2vec_dict, word_dim, reviews):
    # zeros or random vec for unknown word
    unknown_word = np.zeros(word_dim)
    # unknown_word = np.random.rand(1, word_dim)[0]
    data_X = np.zeros((len(reviews), word_dim))
    for i, review in enumerate(reviews):
        words = np.zeros((len(review), word_dim))
        for w, word in enumerate(review):
            words[w] = word2vec_dict.get(word,unknown_word)
        # average func
        data_X[i] = np.mean(words,axis=0)
    return data_X

# return vec for reviews sentence
def text_sent2vec(sent2vec_path):
    data_x = []
    with open(sent2vec_path) as file:
        for line in file:
            if not line:
                continue
            data_x.append([float(s) for s in line.strip().split()])
    return np.array(data_x)

def get_reviews_vec(DATA_PATH):
    train_reviews = DATA_PATH.format("train","reviews")
    train_labels = DATA_PATH.format("train","labels")   
    reviews, labels = file_reader(train_reviews, train_labels)
    #print(labels)
    word2vec_path = "./solution/my_word2vec.txt"
    sent2vec_path = "./solution/sent2vec.train.review.txt"
    word2vec, word_dim = word2vec_reader(word2vec_path)
    result_word2vec = text_word2vec(word2vec, word_dim, reviews)[0,:]
    result_sent2vec = text_sent2vec(sent2vec_path)[0,:]
    print("##### result for Problem 4.1 as following :")
    print("dim word2vec : ", len(result_word2vec))
    print("word2vec result :\n", result_word2vec)
    print("dim sent2vec : ", len(result_sent2vec))
    print("sent2vec result :\n", result_sent2vec)

def model_buider(number_of_labels, word_dim):
    model = Sequential()
    model.add(Dense(number_of_labels, activation='softmax', input_shape=(word_dim,)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_training(DATA_PATH, vec_type = "word2vec"):

    batch_size = 256
    epochs = 15

    train_reviews = DATA_PATH.format("train","reviews")
    train_labels = DATA_PATH.format("train","labels")
    dev_reviews = DATA_PATH.format("dev","reviews")
    dev_labels = DATA_PATH.format("dev","labels")
    test_reviews = DATA_PATH.format("test","reviews")
    test_labels = DATA_PATH.format("test","labels")

    train_revs, train_labels = file_reader(train_reviews, train_labels)
    dev_revs, dev_labels = file_reader(dev_reviews, dev_labels)
    test_revs, test_labels = file_reader(test_reviews, test_labels)

    if vec_type == "sent2vec":
        sent2vec_path = "./solution/sent2vec.{}.review.txt"
        train_datas = text_sent2vec(sent2vec_path.format("train"))
        dev_datas = text_sent2vec(sent2vec_path.format("dev"))
        test_datas = text_sent2vec(sent2vec_path.format("test"))
        
    else:
        word2vec_path = "./solution/my_word2vec.txt"
        word2vec_dict, word_dim = word2vec_reader(word2vec_path)
        train_datas = text_word2vec(word2vec_dict, word_dim, train_revs)
        dev_datas = text_word2vec(word2vec_dict, word_dim, dev_revs)
        test_datas = text_word2vec(word2vec_dict, word_dim, test_revs)

    word_dim = train_datas.shape[1]
    number_of_labels = train_labels.shape[1]
    model = model_buider(number_of_labels, word_dim)
    model.fit(train_datas, train_labels, batch_size=batch_size, epochs=epochs)
    loss, accuracy = model.evaluate(test_datas,test_labels,batch_size=batch_size)
    print("vec_type is : ", vec_type)
    print("\nLoss on test :{}, accuracy: {}\n".format(loss, accuracy))

# -main.py
# -hex07_data
#   -rt-polarity.*.reviews.txt
#   -rt-polarity.*.labels.txt
# -solution
#   -sent2vec.*.review.txt
#   -my_word2vec.txt
if __name__ == '__main__':
    DATA_PATH = "./hex07_data/rt-polarity.{}.{}.txt"

    # problem 4.1
    get_reviews_vec(DATA_PATH)

    # problem 4.2
    model = model_training(DATA_PATH, "sent2vec")
    model = model_training(DATA_PATH)