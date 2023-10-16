import os
import tensorflow as tf

from reasoning_with_vectors.condensed.composition_model import CompositionModel


class DeepEncoder(CompositionModel):

    def __init__(self):
        super().__init__()
        self.model_file = os.path.join(self.model_file_path, DeepEncoder.__name__)

    def init_all_weights(self):
        w1 = tf.compat.v1.get_variable("w1", shape=[2 * self.vector_space_dimension, self.vector_space_dimension],
                                       initializer=self.initializer)
        b1 = tf.compat.v1.get_variable("b1", [1, self.vector_space_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())

        w2 = tf.compat.v1.get_variable("w2", shape=[self.vector_space_dimension, int(self.vector_space_dimension / 2)],
                                       initializer=self.initializer)
        b2 = tf.compat.v1.get_variable("b2", [1, int(self.vector_space_dimension / 2)],
                                       initializer=tf.compat.v1.zeros_initializer())

        w3 = tf.compat.v1.get_variable("w3", shape=[int(self.vector_space_dimension / 2),
                                                    int(self.vector_space_dimension / 4)],
                                       initializer=self.initializer)
        b3 = tf.compat.v1.get_variable("b3", [1, int(self.vector_space_dimension / 4)],
                                       initializer=tf.compat.v1.zeros_initializer())

        w4 = tf.compat.v1.get_variable("w4", shape=[int(self.vector_space_dimension / 4),
                                                    int(self.vector_space_dimension / 2)],
                                       initializer=self.initializer)
        b4 = tf.compat.v1.get_variable("b4", [1, int(self.vector_space_dimension / 2)],
                                       initializer=tf.compat.v1.zeros_initializer())

        w5 = tf.compat.v1.get_variable("w5", shape=[int(self.vector_space_dimension / 2), self.output_layer_dimension],
                                       initializer=self.initializer)
        b5 = tf.compat.v1.get_variable("b5", [1, self.output_layer_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())

        self.weights = {"w1": w1,
                        "b1": b1,
                        "w2": w2,
                        "b2": b2,
                        "w3": w3,
                        "b3": b3,
                        "w4": w4,
                        "b4": b4,
                        "w5": w5,
                        "b5": b5
                        }

    def forwardprop(self, xz, zy, segment_ids):
        w1 = self.weights["w1"]
        b1 = self.weights["b1"]
        w2 = self.weights["w2"]
        b2 = self.weights["b2"]
        w3 = self.weights["w3"]
        b3 = self.weights["b3"]
        w4 = self.weights["w4"]
        b4 = self.weights["b4"]
        w5 = self.weights["w5"]
        b5 = self.weights["b5"]
        xy = tf.concat([xz, zy], 1)
        u = tf.add(tf.matmul(xy, w1), b1)
        a = tf.nn.relu(u)
        u = tf.add(tf.matmul(a, w2), b2)
        a = tf.nn.relu(u)
        u = tf.add(tf.matmul(a, w3), b3)
        a = tf.nn.relu(u)
        a = tf.math.segment_sum(a, segment_ids)
        fire_rate = tf.norm(a, ord=1)
        u = tf.add(tf.matmul(a, w4), b4)
        a = tf.nn.relu(u)
        out = tf.add(tf.matmul(a, w5), b5)
        return [out, fire_rate]

    def forwardprop_predict(self, xz, zy, segment_ids):
        w1 = self.weights["w1"]
        b1 = self.weights["b1"]
        w2 = self.weights["w2"]
        b2 = self.weights["b2"]
        w3 = self.weights["w3"]
        b3 = self.weights["b3"]
        w4 = self.weights["w4"]
        b4 = self.weights["b4"]
        w5 = self.weights["w5"]
        b5 = self.weights["b5"]
        xy = tf.concat([xz, zy], 1)
        u = tf.add(tf.matmul(xy, w1), b1)
        a = tf.nn.relu(u)
        u = tf.add(tf.matmul(a, w2), b2)
        a = tf.nn.relu(u)
        u = tf.add(tf.matmul(a, w3), b3)
        a = tf.nn.relu(u)
        a = tf.math.segment_sum(a, segment_ids)
        u = tf.add(tf.matmul(a, w4), b4)
        a = tf.nn.relu(u)
        out = tf.add(tf.matmul(a, w5), b5)
        return out
