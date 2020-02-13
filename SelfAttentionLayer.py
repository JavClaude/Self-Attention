import keras
from keras import backend as K
from keras.layers import Layer

class SelfAttention(Layer):
    '''

    '''
    def __init__(self,
                 d_a = int,
                 r = int,
                 attention_regularizer_weight = float,
                 return_attention = True,
                 kernel_initializer = 'glorot_normal',
                 kernel_regularizer = None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SelfAttention, self).__init__(**kwargs)
        self.d_a = d_a
        self.r = r
        self.attention_regularizer_weight = attention_regularizer_weight
        self.return_attention = return_attention
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.input_spec = keras.layers.InputSpec(min_ndim = 2 )
        self.supports_masking = True


    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        #First Weight
        self.W1 = self.add_weight(shape = (self.d_a, input_dim),
                                  initializer = self.kernel_initializer,
                                  name = 'W1',
                                  regularizer = self.kernel_regularizer)

        #Second Weight
        self.W2 = self.add_weight(shape=(self.r, self.d_a),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=self.kernel_regularizer)

        self.input_spec = keras.layers.InputSpec(min_ndim = 2, axes = {-1: input_dim})
        self.built = True
        super(SelfAttention, self).build(input_shape)

        #main method
    def call(self, inputs):

        first_linear = self._compute_first_output(inputs)

        attention = self._compute_second_output(first_linear)

        sentence_embedding = self._make_embedding_matrix(inputs, attention)

        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._compute_attention_regularizer(attention))

        if self.return_attention:
            return [sentence_embedding, attention]

        else:
            return sentence_embedding

    def _compute_first_output(self, inputs):

        permute_input = K.permute_dimensions(inputs, pattern = (0, 2, 1))

        output = K.dot(self.W1, permute_input)

        permute_output = K.permute_dimensions(output, pattern = (1, 0, 2)) 

        activation_output = K.tanh(permute_output)

        return activation_output

    def _compute_second_output(self, first_linear):

        output = K.dot(self.W2, first_linear)

        permute_output = K.permute_dimensions(output, pattern = (1, 0, 2)) 

        activation_output = K.softmax(permute_output, axis = 2)

        return activation_output

    def _make_embedding_matrix(self, inputs, second_linear):

        embedding_matrix = K.batch_dot(second_linear, inputs)

        return embedding_matrix

    def _compute_attention_regularizer(self, attention):

        batch_size = K.cast(K.shape(attention)[0], K.floatx())

        input_len = K.shape(attention)[1]

        indices = K.expand_dims(K.arange(0, input_len), axis=0)

        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)

        eye = K.eye(self.r)

        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(attention, K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    def compute_output_shape(self, input_shape):

        assert input_shape and len(input_shape) >= 2

        assert input_shape[-1]

        embedding_matrix_shape = (input_shape[0], self.r, input_shape[-1]) # (batch_size, r, LSTM_units)

        attention_shape = (input_shape[0], self.r, input_shape[-2]) # (batch_size, r, seq_length)

        output_shape = list(input_shape)

        output_shape[-2] = self.r

        output_shape[-1] = input_shape[-1]

        return [embedding_matrix_shape, attention_shape]

    def get_config(self):
        config = {
            'd_a' : self.d_a,
            'r' : self.r,
            'attention_regularizer_weight' : self.attention_regularizer_weight,
            'return_attention' : self.return_attention,
            'kernel_initializer' : self.kernel_initializer,
            'kernel_regularizer' : self.kernel_regularizer,
            }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
