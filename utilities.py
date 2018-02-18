import pickle
import numpy as np
import nltk, re
from nltk.corpus import stopwords

def PickleFiles(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()

def LoadFiles(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff

def CleanText(text, contractions):
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

def BalanceDataframe(df, category=['stars']):
    """
    :param df: pandas.DataFrame
    :param categorical_columns: iterable of categorical columns names contained in {df}
    :return: balanced pandas.DataFrame
    """    
    if category is None or not all([col in df.columns for col in category]):
        raise ValueError('Please provide one or more columns containing categorical variables')

    lowest_count = df.groupby(category).apply(lambda x: x.shape[0]).min()
    df = df.groupby(category).apply( 
        lambda x: x.sample(lowest_count)).drop(category, axis=1).reset_index().set_index('level_1')

    df.sort_index(inplace=True)

    return df


def PadBatch(batch, word2int):
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

def GetBatches(x, y, batch_size):
    # Make sure to not exceed amount of data
    for batch_i in range(0, len(x)//batch_size):
        start = batch_i * batch_size
        end = start+batch_size
        batch_x = x[start:end]
        labels = y[start:end]
        pad_batch_x = np.array(pad_batch(batch_x))
        yield pad_batch_x, labels
        
def GetTestBatches(x, batch_size):
    for batch_i in range(0, len(x)//batch_size):
        start = batch_i * batch_size
        end = start+batch_size
        batch = x[start:end]
        pad_batch_test = np.array(pad_batch(batch))
        yield pad_batch_test