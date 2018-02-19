import tensorflow as tf
import numpy as np

class GRU():
    def __init__(self, batch_size, rnn_size, num_layers, num_classes, word_embedding_matrix, keep_prob):
        self.train_graph = tf.Graph

        with self.train_graph.as_default():
            with tf.name_scope("inputs"):
                self.input_data, self.labels, self.lr, self.keep_prob = self.model_inputs()
                self.weight = tf.Variable(
                    tf.truncated_normal([rnn_size, num_classes], stddev=(1 / np.sqrt(rnn_size * num_classes))))
                self.bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

            self.embeddings = word_embedding_matrix
            self.embs = tf.nn.embedding_lookup(self.embeddings, self.input_data)

            with tf.name_scope("RNN_Layers"):
                stacked_rnn = []
                for layer in range(num_layers):
                    cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                            output_keep_prob=keep_prob)
                    stacked_rnn.append(cell_fw)
                self.multilayer_cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple=True)

            with tf.name_scope("init_state"):
                self.initial_state = self.multilayer_cell.zero_state(batch_size, tf.float32)

            with tf.name_scope("Forward_Pass"):
                self.output, self.final_state = tf.nn.dynamic_rnn(self.multilayer_cell,
                                                                  self.embs,
                                                                  dtype=tf.float32)

            with tf.name_scope("Predictions"):
                self.last = self.output[:, -1, :]
                self.predictions = tf.exp(tf.matmul(last, weight) + bias)
                tf.summary.histogram('predictions', self.predictions)

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
            self.graph_location = "./graph"

            Graph = namedtuple('train_graph', export_nodes)
            local_dict = locals()
            graph = Graph(*[local_dict[each] for each in export_nodes])

            print(graph_location)
            train_writer = tf.summary.FileWriter(graph_location)
            train_writer.add_graph(train_graph)

    def model_inputs(self):
        # Should be [batch_size x review length]
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        # Should be [batch_size x num_classes]
        labels = tf.placeholder(tf.int32, [None, None], name='labels')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return input_data, labels, lr, keep_prob

