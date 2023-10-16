import os
import tensorflow as tf

from reasoning_with_vectors.condensed.composition_model import CompositionModel


class CompositionModel2(CompositionModel):
    def __init__(self):
        super().__init__()
        self.model_file = os.path.join(self.model_file_path, CompositionModel2.__name__)

    def init_all_weights(self):
        W1 = tf.compat.v1.get_variable("W1", shape=[2 * self.vector_space_dimension, self.inner_layer_dimension],
                                       initializer=self.initializer)
        b1 = tf.compat.v1.get_variable("b1", [1, self.inner_layer_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())

        W2 = tf.compat.v1.get_variable("W2", shape=[self.vector_space_dimension, self.inner_layer_dimension],
                                       initializer=self.initializer)
        b2 = tf.compat.v1.get_variable("b2", [1, self.inner_layer_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())

        W3 = tf.compat.v1.get_variable("W3", shape=[self.vector_space_dimension, self.inner_layer_dimension],
                                       initializer=self.initializer)
        b3 = tf.compat.v1.get_variable("b3", [1, self.inner_layer_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())

        W4 = tf.compat.v1.get_variable("W4", shape=[self.vector_space_dimension, self.inner_layer_dimension],
                                       initializer=self.initializer)
        b4 = tf.compat.v1.get_variable("b4", [1, self.inner_layer_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())

        W5 = tf.compat.v1.get_variable("W5", shape=[self.vector_space_dimension, self.inner_layer_dimension],
                                       initializer=self.initializer)
        b5 = tf.compat.v1.get_variable("b5", [1, self.inner_layer_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())

        W6 = tf.compat.v1.get_variable("W6", shape=[3 * self.inner_layer_dimension, self.output_layer_dimension],
                                       initializer=self.initializer)
        b6 = tf.compat.v1.get_variable("b6", [1, self.output_layer_dimension],
                                       initializer=tf.compat.v1.zeros_initializer())
        self.weights = {"W1": W1,
                        "b1": b1,
                        "W2": W2,
                        "b2": b2,
                        "W3": W3,
                        "b3": b3,
                        "W4": W4,
                        "b4": b4,
                        "W5": W5,
                        "b5": b5,
                        "W6": W6,
                        "b6": b6
                        }

    def forwardprop(self, XZ, YZ, segment_ids):
        W1 = self.weights["W1"]
        b1 = self.weights["b1"]
        W2 = self.weights["W2"]
        b2 = self.weights["b2"]
        W3 = self.weights["W3"]
        b3 = self.weights["b3"]
        W4 = self.weights["W4"]
        b4 = self.weights["b4"]
        W5 = self.weights["W5"]
        b5 = self.weights["b5"]
        W6 = self.weights["W6"]
        b6 = self.weights["b6"]
        XY = tf.concat([XZ, YZ], 1)
        U = tf.add(tf.matmul(XY, W1), b1)
        A = tf.nn.gelu(U)
        V = tf.add(tf.matmul(XZ, W2), b2)
        B = tf.nn.gelu(V)
        B = tf.math.multiply(B, tf.add(tf.matmul(YZ, W3), b3))
        V = tf.add(tf.matmul(YZ, W4), b4)
        C = tf.nn.gelu(V)
        C = tf.math.multiply(C, tf.add(tf.matmul(XZ, W5), b5))
        ABC = tf.concat([A, B, C], 1)
        ABC = tf.math.segment_sum(ABC, segment_ids)
        FireRate = tf.norm(ABC, ord=1)
        Output = tf.add(tf.matmul(ABC, W6), b6)
        return [Output, FireRate]

    def forwardprop_predict(self, XZ, YZ, segment_ids):
        W1 = self.weights["W1"]
        b1 = self.weights["b1"]
        W2 = self.weights["W2"]
        b2 = self.weights["b2"]
        W3 = self.weights["W3"]
        b3 = self.weights["b3"]
        W4 = self.weights["W4"]
        b4 = self.weights["b4"]
        W5 = self.weights["W5"]
        b5 = self.weights["b5"]
        W6 = self.weights["W6"]
        b6 = self.weights["b6"]
        XY = tf.concat([XZ, YZ], 1)
        U = tf.add(tf.matmul(XY, W1), b1)
        A = tf.nn.gelu(U)
        V = tf.add(tf.matmul(XZ, W2), b2)
        B = tf.nn.gelu(V)
        B = tf.math.multiply(B, tf.add(tf.matmul(YZ, W3), b3))
        V = tf.add(tf.matmul(YZ, W4), b4)
        C = tf.nn.gelu(V)
        C = tf.math.multiply(C, tf.add(tf.matmul(XZ, W5), b5))
        ABC = tf.concat([A, B, C], 1)
        ABC = tf.math.segment_sum(ABC, segment_ids)
        Output = tf.add(tf.matmul(ABC, W6), b6)
        return Output

