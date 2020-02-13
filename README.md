#### Self Attention (Keras Layer)


##### Sequential Model

For sequential use, parameter **return_attention** must be equal to **False**

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Flatten, Dense
import SelfAttentionLayer

model = Sequential()
model.add(Embedding(len(tokenizer.word_index), 300, input_length = seq_train.shape[1], weights = [embedding_matrix]))
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(SelfAttention(300, 15, attention_regularizer_weights = 0.5, return_attention = False))
model.add(Flatten())
model.add(Dense(2))
```

##### API Model

```python
from keras.layers import Input, Embedding, LSTM, Bidirectional, Flatten, GlobalAveragePooling1D, BatchNormalization, Dense
import SelfAttentionLayer

Inputs = Input(shape=(seq_train.shape[1],))

Embeddings = Embedding(len(tokenizer.vocab), 512, input_length = seq_train.shape[1], weights = [embedding_matrix]) #

Embeded = Embeddings(Inputs)

BatchN1 = BatchNormalization()(Embeded)

BLSTM1 = Bidirectional(LSTM(256, return_sequences=True))(BatchN1)

BatchN2 = BatchNormalization()(BLSTM1)

BLSTM2 = Bidirectional(LSTM(256, return_sequences=True))(BatchN2)

selfAttention = SelfAttention(300, 30, return_attention = True, attention_regularizer_weight=0.5)(BLSTM2)

attention_matrix = selfAttention[1]

Flat = GlobalAveragePooling1D()(selfAttention[0])

Outputs = Dense(2, activation = "sigmoid")(Flat)

```
