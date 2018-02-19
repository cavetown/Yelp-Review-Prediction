import pickle
import numpy as np
import nltk, re
import tensorflow as tf
from nltk.corpus import stopwords


def pickle_files(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


def load_files(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


def clean_text(text, contractions):
    text = text.lower()    
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def pad_batch(batch, word2int):
    # Want to pad this way since tensorflow preprocessing pads with 0's, which can eventually lead to zero tensors
    lengths = []
    for text in batch:
        lengths.append(len(text))
    max_length = max(lengths)
    pad_text = tf.keras.preprocessing.sequence.pad_sequences(batch, 
                                                             maxlen=max_length, 
                                                             padding='post', 
                                                             value=word2int['<pad>'])
    return pad_text


def get_batches(x, y, batch_size, word2int):
    '''
    Getter function for grabbing training batches to be input into the network
    :param x: type list. List of list of reviews to grab from
    :param y: type list corresponding of ratings corresponding to reviews
    :param batch_size: amount of data to feed
    :return: yields training batch reviews with their labels
    '''
    # Make sure to not exceed amount of data
    for batch_i in range(0, len(x)//batch_size):
        start = batch_i * batch_size
        end = start+batch_size
        batch_x = x[start:end]
        labels = y[start:end]
        pad_batch_x = np.asarray(pad_batch(batch_x, word2int))
        yield pad_batch_x, labels


def get_test_batches(x, batch_size):
    '''
    :param x: type list. Train input data to be parsed into batches
    :param batch_size: Batch size hyperparameter
    :return: Yields a test batch for input to neural network
    '''

    for batch_i in range(0, len(x)//batch_size):
        start = batch_i * batch_size
        end = start+batch_size
        batch = x[start:end]
        pad_batch_test = np.array(pad_batch(batch))
        yield pad_batch_test