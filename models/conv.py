import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models._utils import lazy_property, minibatch, load_dataset, DataSet
from sklearn.model_selection import train_test_split

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for Adam Optimizer')
tf.app.flags.DEFINE_float('init_stddev', 0.1, 'Standard deviation for the weight initializer')
tf.app.flags.DEFINE_float('lambda_emb', 0, 'Weight used for l2 regularisation of embedding')
tf.app.flags.DEFINE_float('lambda_conv', 0, 'Weight used for l2 regularisation of convolution')
tf.app.flags.DEFINE_string('logdir', '/tmp/tf-checkpoints/words', 'Logdir to be used')
tf.app.flags.DEFINE_string('datafile', 'data/data.json', 'Datafile to be used')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('epochs', 5, 'Number of training epochs')
tf.app.flags.DEFINE_integer('phrase_length', 300, 'Number of words per training')
tf.app.flags.DEFINE_integer('vocab_size', 4096, 'Size of vocabulary used')
tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Size of embedding used')
FLAGS = tf.app.flags.FLAGS


class ConvTextClassifier(object):

    def __init__(self, vocab_size, max_words=500, embedding_size=100,
                 conv_features=1, conv_words=5, conv_stride=1):
        assert conv_features == 1
        self.vocab_size = vocab_size
        self.max_words = max_words
        self.embedding_size = embedding_size
        self.conv_features = conv_features
        self.conv_words = conv_words
        self.conv_stride = 1

        # tensor with the word indices
        self.word_idx = tf.placeholder(tf.int32, shape=[None, max_words],
                                       name='word_idx')
        self.phrase_lengths = tf.placeholder(tf.int32, shape=[None],
                                           name='phrase_lengths')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='label')

        # hack to ensure initialization
        _ = self.train_step
        _ = self.predict_probabilities

    @staticmethod
    def embedding_layer(x, size_in, size_out, name='embedding'):
        with tf.name_scope(name):
            weights_init = tf.truncated_normal([size_in, size_out],
                                               stddev=FLAGS.init_stddev)
            embedding_weights = tf.Variable(weights_init, name='weights')
            word_vecs = tf.nn.embedding_lookup(embedding_weights, x)

            reg = FLAGS.lambda_emb * tf.nn.l2_loss(embedding_weights, name='l2_reg')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)

            tf.summary.histogram('weights', embedding_weights)
            tf.summary.histogram('activation', word_vecs)

        return word_vecs

    @staticmethod
    def conv1d_layer(x, conv_size, conv_features, conv_stride, name='convolution'):
        embedding_size = x.shape[2].value

        with tf.name_scope(name):
            shape = [conv_size, embedding_size, conv_features]
            weights_init = tf.truncated_normal(shape, stddev=FLAGS.init_stddev)
            conv_weights = tf.Variable(weights_init, name='weights')
            conv_bias = tf.Variable(tf.zeros(conv_features), name='bias')
            y = tf.nn.conv1d(x, conv_weights, stride=conv_stride,
                             padding='SAME', name='y')

            tf.summary.histogram('weights', conv_weights)
            tf.summary.histogram('bias', conv_bias)
            tf.summary.histogram('activation', y)

            reg = FLAGS.lambda_conv * tf.nn.l2_loss(conv_weights, name='l2_reg')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)

        return y

    @staticmethod
    def partial_sum(x, lengths, name='partial_sum'):
        """Computes the mean of x[i, :lengths[i]]"""
        with tf.name_scope(name):
            mask = tf.sequence_mask(lengths, maxlen=x.shape[1].value)
            return tf.reduce_sum(tf.cast(mask, tf.float32) * x, axis=-1)

    @lazy_property
    def predict(self):
        word_vecs = self.embedding_layer(
            x=self.word_idx, size_in=self.vocab_size,
            size_out=self.embedding_size, name='word_embedding')

        word_conv = self.conv1d_layer(
            x=word_vecs, conv_size=self.conv_words, conv_features=self.conv_features,
            conv_stride=self.conv_stride, name='word_conv')

        # FIXME THIS ONLY WORKS FOR conv_features = 1, come up with some
        #
        word_scores = tf.squeeze(word_conv, axis=2, name='word_scores')

        mean_scores = tf.div(self.partial_sum(word_scores, self.phrase_lengths),
                             tf.cast(self.phrase_lengths, tf.float32),
                             name='mean_scores')
        tf.summary.histogram('scores', mean_scores)
        return mean_scores

    @lazy_property
    def predict_probabilities(self):
        return tf.nn.sigmoid(self.predict, name='prob')

    @lazy_property
    def metrics(self):
        logits = self.predict
        with tf.name_scope('metrics'):
            cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(self.labels, logits.dtype), logits=logits))
            tf.summary.scalar('cross_entropy', cross_entropy)

            labels_pred = tf.cast(logits > 0, tf.int32, name='labels_pred')
            correct_prediction = tf.equal(self.labels, labels_pred)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.histogram('labels_pred', labels_pred)
            tf.summary.scalar('accuracy', accuracy)

        return cross_entropy, accuracy

    @property
    def cross_entropy(self):
        cross_entropy, _ = self.metrics
        return cross_entropy

    @property
    def accuracy(self):
        _, accuracy = self.metrics
        return accuracy

    @lazy_property
    def loss(self):
        cross_entropy = self.cross_entropy
        with tf.name_scope('loss'):
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = cross_entropy + sum(reg_losses)
            tf.summary.scalar('reg_cross_entropy', loss)
        return loss

    @lazy_property
    def train_step(self):
        loss = self.loss
        with tf.name_scope('train'):
            train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
        return train


def train(train_set, test_set, model, batch_size):
    init = tf.global_variables_initializer()

    logdir = FLAGS.logdir
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(logdir + '/train')
    test_writer = tf.summary.FileWriter(logdir + '/test')
    train_writer.add_graph(tf.get_default_graph())

    nr_batches = len(train_set.labels) // FLAGS.batch_size + 1
    test_feed = {model.word_idx: test_set.word_idx, model.labels: test_set.labels,
                 model.phrase_lengths: test_set.phrase_lengths}

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(FLAGS.epochs):
            iterator = zip(minibatch(train_set.word_idx, FLAGS.batch_size),
                           minibatch(train_set.labels, FLAGS.batch_size),
                           minibatch(train_set.phrase_lengths, FLAGS.batch_size))
            pbar = tqdm(iterator, total=nr_batches)

            for word_idx, labels, phrase_lengths in pbar:
                feed_dict = {model.word_idx: word_idx, model.labels: labels,
                             model.phrase_lengths: phrase_lengths}
                _, training_loss, training_accuracy = \
                    sess.run([model.train_step, model.loss, model.accuracy],
                             feed_dict=feed_dict)
                pbar.set_description(f'loss: {training_loss}, acc: {training_accuracy}')

            train_summary = sess.run(summary, feed_dict=feed_dict)
            train_writer.add_summary(train_summary, epoch)

            test_loss, test_accuracy, test_summary = \
                sess.run([model.loss, model.accuracy, summary], feed_dict=test_feed)
            test_writer.add_summary(test_summary, epoch)
            print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

            saver.save(sess, save_path=FLAGS.logdir + '/model.ckpt',
                       global_step=epoch)

    print('Done\n\n')


def split_dataset(data, **kwargs):
    result = train_test_split(*data, stratify=data.labels, **kwargs)
    return DataSet(*result[::2]), DataSet(*result[1::2])


def main(_):
    data = load_dataset(FLAGS.datafile, FLAGS.phrase_length,
                        vocab_size=FLAGS.vocab_size)
    assert np.all(data.phrase_lengths > 0)
    data_train, data_test = split_dataset(data, test_size=0.2,
                                          random_state=np.random.RandomState(314))
    model = ConvTextClassifier(vocab_size=FLAGS.vocab_size,
                               max_words=FLAGS.phrase_length,
                               embedding_size=FLAGS.embedding_size)
    train(data_train, data_test, model, FLAGS.batch_size)


if __name__ == '__main__':
    tf.app.run()
