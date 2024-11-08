from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import random
import pickle
app = Flask(__name__)
def importing_vocabularies():
    '''
    This function will load the vocabularies and the index to token dictionaries
    '''
    with open('Necessary_Imports/english_vocab.pkl', 'rb') as f:
        english_vocab = pickle.load(f)

    with open('Necessary_Imports/english_index_to_token.pkl', 'rb') as f:
        english_index_to_token = pickle.load(f)

    with open('Necessary_Imports/french_vocab.pkl', 'rb') as f:
        french_vocab = pickle.load(f)

    with open('Necessary_Imports/french_index_to_token.pkl', 'rb') as f:
        french_index_to_token = pickle.load(f)
    return english_vocab, english_index_to_token, french_vocab, french_index_to_token

english_vocab, english_index_to_token, french_vocab, french_index_to_token = importing_vocabularies()

def translate_to_french(sentence,  french_vocab_inverse = french_index_to_token ,english_vocab = english_vocab,max_src_len = 26):
    '''
    What this function will do is take an English Sentence, convert it to tokens, then convert to indicies, pass these indices to the model, use model.inference to generate the indices of the french sentence, the convert these indices to tokens and finally to a sentence
    sentence: English sentence to translate
    french_vocab_inverse: Inverse of the French vocabulary (index to token)
    english_vocab: English vocabulary (token to index)
    max_src_len: Maximum length of source sentences 
    '''
    model.eval()
    # Tokenize the sentence
    sentence = word_tokenize(sentence.lower())
    
    indices = []
    for token in sentence:
        indices.append(english_vocab.get(token, english_vocab["<UNK>"])) # if token not found, use the index of the <UNK> token
    # Get the lengths of the sequences, before padding 
    lengths = torch.tensor([len(indices)]) # shape: (1)

    # Pad the sequence
    if len(indices) < max_src_len:
        indices += [english_vocab["<PAD>"]] * (max_src_len - len(indices))
    # Cut off the sequence if it exceeds the maximum length
    elif len(indices) > max_src_len:
        indices = indices[:max_src_len]
    indices = torch.tensor(indices).unsqueeze(0) # shape: (1, max_src_len)
    with torch.no_grad():
        translated_indices = model.inference(indices, lengths)
    # Convert the indices to tokens
    translated_tokens = []
    for index in translated_indices[0]:
        token = french_vocab_inverse[index.item()]
        if token == "<EOS>":
            break
        translated_tokens.append(token)
    # Join the tokens to form the translated sentence
    return ' '.join(translated_tokens)

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=2):
        '''
        input_dim: The size of the input vocabulary (It tells the model how many unique tokens it can expect to see)
        '''
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim) # it will have a matrix of size (input_dim, embedding_dim)
        # where each row of the matrix will represent the embedding of a token in the vocabulary
        '''Now each token in my input sequence is an index that refers to a row in this embedding matrix.'''
        '''This lookup returns a new tensor of shape (batch_size, max_length, embedding_dim)'''
        
        '''Modification: Make the LSTM bidirectional by setting bidirectional=True'''
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True) # batch_first=True means that the first dimension of the input and output will be the batch size (batch_size, max_length, hidden_dim)
        # note that in LSTM, the time step happens under the hood
        # LSTM treats each position along sequence length as a time step

    def forward(self, src, src_lengths):
        # src shape: (batch_size, max_length) - a batch of sequences of token indices
        # src_lengths shape: (batch_size) - the actual lengths of each sequence without padding

        '''If src has shape (batch_size, max_length) (a batch of padded sequences of token indices), the embedding layer’s output will be a tensor of shape (batch_size, max_length, embedding_dim)'''
        embedded = self.embedding(src)  # shape: (batch_size, max_length, embedding_dim)
        

        '''LSTM will process only the non-padded elements'''
        packed_embedded = pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=False) 
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        # hidden and cell shapes: (num_layers * num_directions, batch_size, hidden_dim)
        
        '''Modification: Since the LSTM is bidirectional, we need to handle the hidden and cell states accordingly.
        The hidden and cell states from the bidirectional LSTM have shapes:
        - hidden: (num_layers * num_directions, batch_size, hidden_dim)
        We need to combine the hidden states from both directions to pass to the decoder.
        One common approach is to sum the hidden states from both directions.
        '''
        # Reshape hidden and cell to (num_layers, num_directions, batch_size, hidden_dim)
        num_layers = self.lstm.num_layers
        num_directions = 2  # Because bidirectional=True
        hidden = hidden.view(num_layers, num_directions, hidden.size(1), hidden.size(2))
        cell = cell.view(num_layers, num_directions, cell.size(1), cell.size(2))

        # Sum the hidden states from both directions
        hidden = hidden.sum(dim=1)  # Now shape: (num_layers, batch_size, hidden_dim)
        cell = cell.sum(dim=1)      # Now shape: (num_layers, batch_size, hidden_dim)

        # Note if you need the outputs you need to unpack the sequences using pad_packed_sequence
        # outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True) #outputs shape: (batch_size, max_length, hidden_dim * num_directions)
    
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers=2, teacher_forcing_ratio=0.5):
        super(Decoder, self).__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Embedding layer for the output vocabulary( French vocabulary)
        self.embedding = nn.Embedding(output_dim, embedding_dim)
    
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer to generate predictions over vocabulary
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_token, hidden, cell):
        # input_token shape: (batch_size) - single token for each sequence in the batch, since later in the full model we will be looping over the sequence length(1 token at a time)
        
        '''Step 1: Embed the input token'''
        embedded = self.embedding(input_token).unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)
        
        '''Step 2: Pass the embedded input through the LSTM'''
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # Shape: (batch_size, 1, hidden_dim)
        
        '''Step 3: Generate prediction for the token'''
        prediction = self.fc_out(output.squeeze(1))  # Shape: (batch_size, output_dim)
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,  teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
    def forward(self, src, src_lengths, tgt):
        batch_size = src.shape[0]
        tgt_sequence_length = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features

        # Initialize tensor to hold decoder outputs
        outputs = torch.zeros(batch_size, tgt_sequence_length - 1, tgt_vocab_size)

        # Encode the source sequence
        hidden, cell = self.encoder(src, src_lengths)

        # The first input to the decoder is the <SOS> token
        input_token = tgt[:, 0]  # Shape: (batch_size)

        for t in range(1, tgt_sequence_length):
            # Pass the input token through the decoder
            output, hidden, cell = self.decoder(input_token, hidden, cell)

            # Store the output prediction
            outputs[:, t - 1, :] = output

            # Decide whether to use teacher forcing
            teacher_force = random.random() < self.teacher_forcing_ratio

            # Get the next input token
            top1 = output.argmax(1)

            input_token = tgt[:, t] if teacher_force else top1

        return outputs
    def inference(self, src, src_lengths, max_len=26, start_token=1, end_token=2):
        '''Will output the generated indices of the target sequence'''
        # src: Source input sequence - Shape: (batch_size, src_sequence_length) (English)
        # src_lengths: Actual lengths of each sequence in src (ignores padding)
        # max_len: Maximum length of the generated sequence 
        # start_token: Start-of-sequence token index
        # end_token: End-of-sequence token index
        
        '''Will output the generated indices of the target sequence'''
        batch_size = src.shape[0]

        # Encode the source sequence
        hidden, cell = self.encoder(src, src_lengths)

        # Initialize the input token with the <SOS> token
        input_token = torch.tensor([start_token] * batch_size)  # Shape: (batch_size)

        # Initialize tensor to hold generated tokens
        generated_tokens = torch.zeros(batch_size, max_len).long()

        for t in range(max_len):
            # Pass the input token through the decoder
            output, hidden, cell = self.decoder(input_token, hidden, cell)

            # Get the predicted token
            predicted_token = output.argmax(1)  # Shape: (batch_size)
            generated_tokens[:, t] = predicted_token

            # Check if all sequences have predicted <EOS>
            if (predicted_token == end_token).all():
                break

            # Update input token for the next time step
            input_token = predicted_token

        return generated_tokens


def setting_up_model(path = 'biderctional_model/seq2seq_model_8.pth'):
    '''
    This function will load the model and the vocabularies
    '''
    input_dim = len(english_vocab)  
    output_dim = len(french_vocab)  
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 3
    encoder = Encoder(input_dim, embedding_dim, hidden_dim, num_layers)
    decoder = Decoder(output_dim, embedding_dim, hidden_dim, num_layers)
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(torch.load(path,weights_only=True))
    model.eval()
    return model
model = setting_up_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    original_sentence = ""
    translated_sentence = ""
    
    if request.method == 'POST':
        original_sentence = request.form['sentence']  # Capture the original sentence
        translated_sentence = translate_to_french(original_sentence)  # Perform the translation

    return render_template('index.html', original_sentence=original_sentence, translated_sentence=translated_sentence)

if __name__ == '__main__':
    app.run(debug=True)