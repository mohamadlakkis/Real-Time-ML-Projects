import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


train_df = pd.read_csv('train.csv')
def preprocess_dataset(df):
    # Replace special values in the "Input" column, to "NULL" to stay consistent
    df['input'] = df['input'].replace(["Not applicable", "NULL", "INPUT","",np.NAN], "NULL")
    df['output'] = df['output'].astype(str)
    df['input'] = df['input'].astype(str)
    df['instruction'] = df['instruction'].astype(str)
    return df
train_df = preprocess_dataset(train_df)
train_df.to_csv('train_preprocessed.csv', index=False)


instructions = train_df['instruction'].values
inputs = train_df['input'].values
outputs = train_df['output'].values
inputs  = [instructions[i] + " " + inputs[i] for i in range(len(instructions))]


tokenizer = Tokenizer(oov_token = '<OOV>')
tokenizer.fit_on_texts(inputs + outputs)
input_sequences = tokenizer.texts_to_sequences(inputs)
output_sequences = tokenizer.texts_to_sequences(outputs)


max_len_input = max(len(seq) for seq in input_sequences)
max_len_output = max(len(seq) for seq in output_sequences)
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
padded_input_sequences = pad_sequences(input_sequences, maxlen=vocab_size, padding='post')
padded_output_sequences = pad_sequences(output_sequences, maxlen=vocab_size, padding='post')


                        
vocab = tokenizer.word_index

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64), # , input_length=max_len_input is no longer necessary it will automatically detect it
    SimpleRNN(128, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

print(padded_input_sequences.shape)
print(padded_output_sequences.shape)
history = model.fit(
    input_sequences,
    output_sequences,
    batch_size=8,
    epochs=1,
    validation_split=0.2,
    verbose=1
)


