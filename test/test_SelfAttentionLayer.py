import unittest
import tensorflow as tf
import keras.backend as K

from . import SelfAttentionLayer

class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.attentionLayer = SelfAttention(d_a = 150, r = 10, attention_regularizer = 0.5, return_attention = False)
        self.batch_size_for_test = -1
        self.max_seq_for_test = 150
        self.encoder_units_for_test = 256
        self.encoder_output = K.random.normal([self.batch_size_for_test, self.max_seq_for_test, self.encoder_units_for_test])
        print('Input shape for test: ' + str([self.batch_size_for_test, self.max_seq_for_test, self.encoder_units_for_test]))


class TestInit(TestSelfAttention):

    def test_initial_d_a(self):
        self.assertIsNotNone(self.attentionLayer.d_a)

    def test_initial_r(self):
        self.assertIsNotNone(self.attentionLayer.r)

    def test_initial(self):
        self.assertIsNotNone(self.attentionLayer.attention_regularizer_weight)


class TestTensorDot(TestSelfAttention):

    def test_dot_product(self):
        self.attentionLayer.build([self.batch_size_for_test, self.max_seq_for_test, self.encoder_units_for_test])

        first_output = self.attentionLayer._compute_first_output(self.encoder_output)

        first_output_shape = first_output.get_shape().as_list()

        attention_matrix = self.attentionLayer._compute_second_output(first_output)

        attention_matrix_shape = attention_matrix.get_shape().as_list()

        embedding_matrix = self.attentionLayer._make_embedding_matrix(self.encoder_output, attention_matrix)

        embedding_matrix_shape = embedding_matrix.get_shape().as_list()
        # Matrix [batch size, d_a, max_seq]
        self.assertEqual(first_output_shape, [None, self.attentionLayer.d_a, self.max_seq_for_test], "Error on first tensor product")
        # Attention matrix [batch size, r, max_seq]
        self.assertEqual(attention_matrix_shape, [None, self.attentionLayer.r, self.max_seq_for_test], "Error on second tensor product")
        # Embedding matrix [batch size, r, LSTM units]
        self.assertEqual(embedding_matrix_shape, [None, self.attentionLayer.r, self.encoder_units_for_test], "Error on third tensor product")
