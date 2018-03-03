from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
from tokenizer import Tokenizer
import embedding_utils as emb_utils
import utilities as utils
import argparse
from contractions import get_contractions
# Working with TensorFlow v1.6
print('TensorFlow Version: {}'.format(tf.__version__))

parser = argparse.ArgumentParser(description="Specify number of reviews to parse")
parser.add_argument("-nl", "--num_layers", type=int, default=2, help="Specify number of layers for GRU RNN")
parser.add_argument("-bz", "--batch_size", type=int, default=64, help="Batch size to train network")
parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs to train")
parser.add_argument("-hu", "--hidden_units", type=int, default=64, help="number of hidden units for the network")
parser.add_argument("-t", "--task", type=str, default='train',
                    help="Specify whether to train the network, predict, or  test")
parser.add_argument("-kp", "--keep_prob", type=float, default=0.8, help="amount to keep during dropout")
parser.add_argument("-sl", "--max_sequence_length", type=int, default=750, help="Max length of a sequence to be trained")
parser.add_argument("-ep", "--embedding_path", type=str, default='./embeddings/numberbatch-en.txt',
                    help="Path to embedding matrix")
parser.add_argument("-f", "--file", type=str, default="./balanced_reviews.csv", help="Path to csv of preprocessed data")
parser.add_argument("-ed", "--embedding_dim", type=int, default=300,
                    help="Number of dimensions for your embedding matrix")
parser.add_argument("-v", "--val_split", type=float, default=0.2, help="Amount of data to use for validation")
parser.add_argument("-p", "--pickle", type=str, default='true', help="Specify whether to pickle files")
parser.add_argument("-r", "--resume", type=str, default='false', help="Resume training")
parser.add_argument("-lrd", "--learning_rate_decay", type=float, default=0.95, help="Fraction of LR to keep every time")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.005, help="Learning Rate")
parser.add_argument("-s", "--shuffle", type=str, default='false', help="shuffle data after each epoch")
parser.add_argument("-uc", "--update_check", type=int, default=500,
                    help="how many batches to check for updates in training")
args = parser.parse_args()

NUM_CLASSES = 6


def model_inputs():
    # Should be [batch_size x review length]
    inp = tf.placeholder(tf.int32, [None, None], name='input')
    # Should be [batch_size x num_classes]
    target = tf.placeholder(tf.int32, [None, None], name='labels')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_probability = tf.placeholder(tf.float32, name='keep_prob')
    return inp, target, learning_rate, keep_probability


# Load main dataframe
df_balanced = pd.read_csv(args.file)
# Load tokenizer class
tokenizer = Tokenizer()
# Load Embeddings matrix
embeddings_index = emb_utils.load_embeddings(args.embedding_path)
# Have tokenizer fit to our data
tokenizer.fit_on_texts(df_balanced.text, embeddings_index)

word_embedding_matrix = emb_utils.create_embedding_matrix(tokenizer.word2int, embeddings_index, args.embedding_dim)
seq = tokenizer.text_to_sequence(df_balanced['text'])

# Creating graph for TensorFlow
tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    with tf.name_scope("inputs"):
        input_data, labels, lr, keep_prob = model_inputs()
        weight = tf.Variable(
            tf.truncated_normal([args.hidden_units, NUM_CLASSES], stddev=(1 / np.sqrt(args.hidden_units * NUM_CLASSES))))
        bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))

    embeddings = word_embedding_matrix
    embs = tf.nn.embedding_lookup(embeddings, input_data)

    with tf.name_scope("RNN_Layers"):
        stacked_rnn = []
        for layer in range(args.num_layers):
            cell_fw = tf.contrib.rnn.GRUCell(args.hidden_units)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    output_keep_prob=keep_prob)
            stacked_rnn.append(cell_fw)
        multilayer_cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple=True)

    with tf.name_scope("init_state"):
        initial_state = multilayer_cell.zero_state(args.batch_size, tf.float32)

    with tf.name_scope("Forward_Pass"):
        output, final_state = tf.nn.dynamic_rnn(multilayer_cell,
                                                embs,
                                                dtype=tf.float32)

    with tf.name_scope("Predictions"):
        last = output[:, -1, :]
        predictions = tf.exp(tf.matmul(last, weight) + bias)
        tf.summary.histogram('predictions', predictions)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels))
        tf.summary.scalar('cost', cost)

    # Optimizer
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    # Predictions comes out as 6 output layer, so need to "change" to one hot
    with tf.name_scope("accuracy"):
        correctPred = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    export_nodes = ['input_data', 'labels', 'keep_prob', 'lr', 'initial_state', 'final_state',
                    'accuracy', 'predictions', 'cost', 'optimizer', 'merged']

    merged = tf.summary.merge_all()

print("Graph is built.")
graph_location = "./graph"

Graph = namedtuple('train_graph', export_nodes)
local_dict = locals()
graph = Graph(*[local_dict[each] for each in export_nodes])

print(graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(train_graph)


def train(x_train, y_train, batch_size, keep_probability, learning_rate, display_step=20, update_check=500):
    """
    Main train file to run the code. It will take in the parameters and train the network and then save the model
    every some iterations.

    :param x_train: input to the network
    :type x_train: list
    :param y_train: labels for the network
    :type y_train: list
    :param batch_size: How large of a set of inputs to feed into network
    :type batch_size: int
    :param keep_probability: How much of dropout to use
    :type keep_probability: float
    :param learning_rate: Learning rate for the network
    :type learning_rate: float
    :param display_step: How often to display updates onto the screen (how many iterations)
    :type display_step: int
    :param update_check: How often to check to save model (number of times per epoch)
    :type update_check: int
    :return: None
    """
    print("Training Now")
    epochs = args.epochs
    summary_update_loss = []
    min_learning_rate = 0.0005
    stop_early = 0
    stop = 3
    checkpoint = "./saves/best_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        if args.resume == 'true':
            loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
            loader.restore(sess, checkpoint)
        else:
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter('./summaries' + '/train', sess.graph)

        for epoch_i in range(1, epochs + 1):
            state = sess.run(graph.initial_state)

            update_loss = 0
            batch_loss = 0

            for batch_i, (x, y) in enumerate(utils.get_batches(x_train, y_train, batch_size, tokenizer.word2int)):
                if batch_i == 1 and epoch_i == 1:
                    print("Starting")
                feed = {graph.input_data: x,
                        graph.labels: y,
                        graph.keep_prob: keep_probability,
                        graph.initial_state: state,
                        graph.lr: learning_rate}
                start_time = time.time()
                summary, loss, acc, state, _ = sess.run([graph.merged,
                                                         graph.cost,
                                                         graph.accuracy,
                                                         graph.final_state,
                                                         graph.optimizer],
                                                        feed_dict=feed)
                if batch_i == 1 and epoch_i == 1:
                    print("Finished first")

                train_writer.add_summary(summary, epoch_i * batch_i + batch_i)

                batch_loss += loss
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Acc: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(x_train) // batch_size,
                                  batch_loss / display_step,
                                  acc,
                                  batch_time * display_step))
                    batch_loss = 0

                if batch_i % update_check == 0 and batch_i > 0:
                    print("Average loss for this update:", round(update_loss / update_check, 3))
                    summary_update_loss.append(update_loss)

                    # If the update loss is at a new minimum, save the model
                    if update_loss <= min(summary_update_loss):
                        print('New Record!')
                        stop_early = 0
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break
                    update_loss = 0

            # Reduce learning rate, but not below its minimum value
            learning_rate *= args.learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            # Set shuffle to True if you want to shuffle data between epochs
            # This can add some randomness and potentially learn new patterns in data
            # if args.shuffle.lower() == 'true':
            #     x_train, y_train = utils.shuffle_data(x_train, y_train)
            if stop_early == stop:
                print("Stopping Training.")
                break
    print("Done Training")


def test(x_test, y_test):
    """
    Tests the network to see how well the network has trained

    :param x_test: input to the test function
    :type x_test: list
    :param y_test: labels for the test function
    :type y_test: list
    :return: None
    """
    print("Testing Now")
    with tf.Session(graph=train_graph) as sess:
        checkpoint = "./saves/best_model.ckpt"
        all_preds = []

        # with tf.Session() as sess:
        saver = tf.train.Saver()
        # Load the model
        saver.restore(sess, checkpoint)
        state = sess.run(graph.initial_state)
        print("Total Batches: %d" % (len(x_test)//args.batch_size))
        for ii, x in enumerate(utils.get_test_batches(x_test, args.batch_size), 1):
            if ii % 100 == 0:
                print("%d batches" % ii)
            feed = {graph.input_data: x,
                    graph.keep_prob: args.keep_prob,
                    graph.initial_state: state}

            test_preds = sess.run(graph.predictions, feed_dict=feed)

            for i in range(len(test_preds)):
                all_preds.append(test_preds[i,:])

    all_preds = np.asarray(all_preds)
    y_predictions = np.argmax(all_preds, axis=1)
    y_true = y_test.argmax(axis=1)
    y_true = y_true[:y_predictions.shape[0]]

    cm = ConfusionMatrix(y_true, y_predictions)
    cm.plot(backend='seaborn', normalized=True)
    plt.title('Confusion Matrix Stars prediction')
    plt.figure(figsize=(12, 10))

    test_correct_pred = np.equal(y_predictions, y_true)
    test_accuracy = np.mean(test_correct_pred.astype(float))

    print("Test accuracy is: " + str(test_accuracy))


def predict():
    pred_text = input("Please enter a review in english")
    contractions = get_contractions()
    pred_text = utils.clean_text(pred_text, contractions)
    pred_seq = tokenizer.text_to_sequence(pred_text, pred=True)
    pred_seq = np.tile(pred_seq, (args.batch_size, 1))

    with tf.Session(graph=train_graph) as sess:
        checkpoint = "./saves/best_model.ckpt"
        all_preds = []
        # with tf.Session() as sess:
        saver = tf.train.Saver()
        # Load the model
        saver.restore(sess, checkpoint)
        state = sess.run(graph.initial_state)
        feed = {graph.input_data: pred_seq,
                graph.keep_prob: args.keep_prob,
                graph.initial_state: state}

        preds = sess.run(graph.predictions, feed_dict=feed)
        for i in range(len(preds)):
            all_preds.append(preds[i, :])
    all_preds = np.asarray(all_preds)
    y_predictions = np.argmax(all_preds, axis=1)
    counts = np.bincount(y_predictions)
    print("\nYou rated the restaurant: " + str(np.argmax(counts)) + " stars!")


def main():
    ratings = df_balanced.stars.values.astype(int)
    ratings_cat = tf.keras.utils.to_categorical(ratings)
    x_train, x_test, y_train, y_test = train_test_split(seq, ratings_cat, test_size=0.2, random_state=9)
    if args.pickle:
        utils.pickle_files("./data/pickles/balanced_reviews.p", df_balanced)
        utils.pickle_files("./data/pickles/category_ratings.p", ratings_cat)
        utils.pickle_files("./data/pickles/word_embedding_matrix.p", word_embedding_matrix)
        utils.pickle_files("./data/pickles/tokenizer.p", tokenizer)

    if args.task == 'train':
        train(x_train, y_train, args.batch_size, args.keep_prob, args.learning_rate, update_check=args.update_check)
        # test(x_test, y_test)
    elif args.task == 'test':
        test(x_test, y_test)
    elif args.task == 'predict':
        predict()


if __name__ == "__main__":
    main()