import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize

from reasoning_with_vectors.conf import configuration


class CompositionModel:
    def __init__(self):
        tf.compat.v1.disable_eager_execution()
        self.vector_space_dimension = configuration.vector_space_dimension
        self.inner_layer_dimension = configuration.inner_layer_dimension
        self.output_layer_dimension = configuration.vector_space_dimension
        self.initializer = tf.keras.initializers.GlorotUniform(42)
        self.model_file_path = configuration.model_save_path
        self.model_file = os.path.join(self.model_file_path, CompositionModel.__name__ +
                                       str(configuration.importance_threshold))
        np.random.seed(42)
        self.xz = None
        self.zy = None
        self.segment_ids = None
        self.xy = None
        self.xy_hat = None
        self.xy_hat_pred = None
        self.fire_rate = None
        self.weights = None
        self.loss = None
        self.updates = None
        self.saver = None
        self.sess = None
        self.init_model()

    def init_model(self):
        self.init_all_weights()
        self.xz, self.zy, self.segment_ids, self.xy = self.create_placeholders()
        [self.xy_hat, self.fire_rate] = self.forwardprop(self.xz, self.zy, self.segment_ids)
        self.xy_hat_pred = self.forwardprop_predict(self.xz, self.zy, self.segment_ids)
        self.loss = tf.reduce_sum(tf.keras.losses.cosine_similarity(self.xy, self.xy_hat))
        self.updates = tf.compat.v1.train.AdamOptimizer(configuration.learning_rate).minimize(self.loss)
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def run_session(self, xz_list, zy_list, segmentid_list, xy_list):
        loss_value = None
        fire_rate_value = None
        xz_train = np.vstack(xz_list)
        xz_train = normalize(xz_train)
        zy_train = np.vstack(zy_list)
        zy_train = normalize(zy_train)
        segment_ids = np.hstack(segmentid_list)
        xy_train = np.vstack(xy_list)
        xy_train = normalize(xy_train)
        for i in range(10):
            [_, loss_value, fire_rate_value] = self.sess.run([self.updates, self.loss, self.fire_rate],
                                                             feed_dict={self.xz: xz_train, self.zy: zy_train,
                                                                        self.segment_ids: segment_ids,
                                                                        self.xy: xy_train})
        logging.info("Loss value = %s, Fire rate value = %s, Number of concept pairs for training = %d",
                     str(loss_value), str(fire_rate_value), len(xy_list))

    def init_all_weights(self):
        w1 = tf.compat.v1.get_variable("w1", shape=[2 * self.vector_space_dimension, self.inner_layer_dimension],
                                       initializer=self.initializer)
        b1 = tf.compat.v1.get_variable("b1", [1, self.inner_layer_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())
        w2 = tf.compat.v1.get_variable("w2", shape=[self.inner_layer_dimension, self.output_layer_dimension],
                                       initializer=self.initializer)
        b2 = tf.compat.v1.get_variable("b2", [1, self.output_layer_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())
        self.weights = {"w1": w1,
                        "b1": b1,
                        "w2": w2,
                        "b2": b2
                        }

    def forwardprop(self, xz, zy, segment_ids):
        w1 = self.weights["w1"]
        b1 = self.weights["b1"]
        w2 = self.weights["w2"]
        b2 = self.weights["b2"]
        xy = tf.concat([xz, zy], 1)
        u = tf.add(tf.matmul(xy, w1), b1)
        a = tf.nn.relu(u)
        a = tf.math.segment_sum(a, segment_ids)
        fire_rate = tf.norm(a, ord=1)
        out = tf.add(tf.matmul(a, w2), b2)
        return [out, fire_rate]

    def forwardprop_predict(self, xz, zy, segment_ids):
        w1 = self.weights["w1"]
        b1 = self.weights["b1"]
        w2 = self.weights["w2"]
        b2 = self.weights["b2"]
        xy = tf.concat([xz, zy], 1)
        u = tf.add(tf.matmul(xy, w1), b1)
        a = tf.nn.relu(u)
        a = tf.math.segment_sum(a, segment_ids)
        out = tf.add(tf.matmul(a, w2), b2)
        return out

    def create_placeholders(self):
        xz = tf.compat.v1.placeholder(tf.float32, shape=[None, self.vector_space_dimension])
        zy = tf.compat.v1.placeholder(tf.float32, shape=[None, self.vector_space_dimension])
        segment_ids = tf.compat.v1.placeholder(tf.int32, shape=[None])
        xy = tf.compat.v1.placeholder(tf.float32, shape=[None, self.output_layer_dimension])
        return xz, zy, segment_ids, xy

    def train_model(self):
        pass

    def save_model(self):
        self.saver.save(self.sess, self.model_file)

    def load_model(self):
        self.sess = tf.compat.v1.Session()
        new_saver = tf.compat.v1.train.import_meta_graph(self.model_file + '.meta')
        new_saver.restore(self.sess, self.model_file)

    def predict(self, xz_list_test, zy_list_test, test_segment_ids):
        xz_test = np.vstack(xz_list_test)
        xz_test = normalize(xz_test)
        zy_test = np.vstack(zy_list_test)
        zy_test = normalize(zy_test)
        test_segment_ids = np.hstack(test_segment_ids)
        prediction = self.sess.run(self.xy_hat_pred, feed_dict={self.xz: xz_test,
                                                                self.zy: zy_test,
                                                                self.segment_ids: test_segment_ids})
        return prediction.tolist()
