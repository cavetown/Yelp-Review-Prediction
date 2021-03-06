import pickle
import numpy as np
import nltk, re
import tensorflow as tf
from nltk.corpus import stopwords


def pickle_files(filename, stuff):
    """
    Saves files to be used later
    :param filename: name to called pickled file
    :type filename: str
    :param stuff: file to pickle
    :return: None
    """
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


def load_files(filename):
    """
    Loads a pickles file
    :param filename: file to load
    :type filename: str
    :return: Loaded file
    """
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


def clean_text(text, contractions):
    """
    Method to clean up text by removing stopwords, splitting, removing punctuation, and none usable
    characters

    :param text: string of text to clean
    :type text: str
    :param contractions: expand contractions so words will be recognized as the same
    :type contractions: dict
    :return: text as a list
    :rtype: list
    """
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
    """
    Pad the batch so that they are all of equal length to be fed into the network

    :param batch: Batch to pad
    :type batch: list
    :param word2int: dictionary to tokenize
    :type word2int: dict
    :return:padded text
    :rtype: list
    """
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
    """
    Getter function for grabbing training batches to be input into the network
    :param x: type list. List of list of reviews to grab from
    :type x: list
    :param y: type list corresponding of ratings corresponding to reviews
    :type y: list
    :param batch_size: amount of data to feed
    :return: yields training batch reviews with their labels
    """
    # Make sure to not exceed amount of data
    for batch_i in range(0, len(x)//batch_size):
        start = batch_i * batch_size
        end = start+batch_size
        batch_x = x[start:end]
        labels = y[start:end]
        pad_batch_x = np.asarray(pad_batch(batch_x, word2int))
        yield pad_batch_x, labels


def get_test_batches(x, batch_size, word2int):
    """
    :param x: type list. Train input data to be parsed into batches
    :type x: list
    :param batch_size: Batch size hyperparameter
    :type batch_size: int
    :param word2int: tokenizing dictionary
    :type word2int: dict
    :return: Yields a test batch for input to neural network
    """

    for batch_i in range(0, len(x)//batch_size):
        start = batch_i * batch_size
        end = start+batch_size
        batch = x[start:end]
        pad_batch_test = np.asarray(pad_batch(batch, word2int))
        yield pad_batch_test
