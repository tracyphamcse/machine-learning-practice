import tensorflow as tf
import time
import math
class Model:
    def __init__(self, vocab_size, embedding_dim = 128, batch_size = 128, num_sampled = 1, stddev = 1.0, learning_rate = 1.0):

        '''
            Placeholder for input and output
            Use embedding_lookup to lookup embedding value of input
        '''
        self.x = tf.placeholder(tf.int32, shape=[batch_size])
        self.y_label = tf.placeholder(tf.int32, shape=[batch_size, 1])


        '''
            Embedding layer using embedding_lookup function
        '''
        embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.x)


        '''
            Hidden layer
        '''

        weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_dim],stddev=stddev / math.sqrt(embedding_dim)))
        biases = tf.Variable(tf.zeros([vocab_size]))

        hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases


        '''
            Output layer
            Using Noise Contrastive Estimation (NCE) loss function
        '''
        nce_weights = tf.Variable(
                tf.truncated_normal([vocab_size, embedding_dim],
                                    stddev=stddev / math.sqrt(embedding_dim)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))
        nce_out = tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=self.y_label,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocab_size)

        '''
            Define the training process
            Using reduce_mean loss and GradientDescentOptimizer
        '''

        self.nce_loss = tf.reduce_mean(nce_out)
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.nce_loss)

        '''
            Normalize embeddings
        '''
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self.normalized_embeddings = embeddings / norm

        '''
            Final embeddings value
        '''
        self.final_embeddings = None

    def run(self, x_train_batches, y_train_batches, n_inters):
        self.sess = tf.Session()

        '''
            Init the global variables for tensorflow
            Depend on the tenfsorflow version, choose the init function name
        '''
        # init = tf.global_variables_initializer()
        init = tf.initialize_all_variables()
        self.sess.run(init)

        '''
            Run the training over minibatches
            with N_ITERS turn

        '''

        num_of_batches = len(x_train_batches)

        print "Number of batches to train", num_of_batches
        print "Number of interations to train", n_inters

        err = 0
        start = time.time()

        for i in range(n_inters):

            _, loss = self.sess.run([self.train_step, self.nce_loss], feed_dict={self.x: x_train_batches[i%num_of_batches], self.y_label: y_train_batches[i%num_of_batches]})

            err = err + loss
            if (i + 1 ) % 1000 == 0 :
                end = time.time()
                print 'avg loss after', i,'turn is : ', err/float(1000), '- in:', (end - start)
                err = 0
                start = time.time()

        self.final_embeddings = self.normalized_embeddings.eval(session=self.sess)

        print "DONE"

    def get_final_embedding(self):
        return self.final_embeddings
