import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
nltk.download('punkt')
nltk.download('punkt_tab')
import string
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import random

en = pd.read_csv('Datasets/en.csv')
fr = pd.read_csv('Datasets/fr.csv')
english_sentences = en.iloc[:, 0]
french_sentences = fr.iloc[:, 0]
en.columns = ['English']
fr.columns = ['French']
combined_df = pd.concat([en, fr], axis=1)
'''remove the punctuation'''
combined_df['English'] = combined_df['English'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
combined_df['French'] = combined_df['French'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

''' convert to lowercase'''
combined_df['English'] = combined_df['English'].apply(lambda x: x.lower())
combined_df['French'] = combined_df['French'].apply(lambda x: x.lower())

combined_df['ENG Length'] = combined_df['English'].apply(lambda x: len(x.split()))
combined_df['FR Length'] = combined_df['French'].apply(lambda x: len(x.split()))

def tokenize_sentences(sentences):
    return [word_tokenize(sentence.lower()) for sentence in sentences]
english_sentences = combined_df['English']
french_sentences = combined_df['French']
english_tokenized = tokenize_sentences(english_sentences)
french_tokenized = tokenize_sentences(french_sentences)


def build_vocabulary(tokenized_sentences, special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"]):
    all_tokens_list = []
    for sentence in tokenized_sentences:
        all_tokens_list.extend(sentence)
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for token in all_tokens_list:
        if token not in vocab:
            vocab[token] = len(vocab) # updating each token with a unique index, since len(vocab) is changing each time
    # Decoding Later 
    index_to_token = {idx: token for token, idx in vocab.items()}
    return vocab, index_to_token
english_vocab, english_index_to_token = build_vocabulary(english_tokenized)
french_vocab, french_index_to_token = build_vocabulary(french_tokenized)

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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True) # batch_first=True means that the first dimension of the input and output will be the batch size (batch_size, max_length, hidden_dim)
        # note that in LSTM, the time step happens under the hood
        # LSTM treats each position along sequence lenth as a time step
        
    def forward(self, src, src_lengths):
        # src shape: (batch_size, max_length) - a batch of sequences of token indices
        # src_lengths shape: (batch_size) - the actual lengths of each sequence without padding


        '''If src has shape (batch_size, max_length) (a batch of padded sequences of token indices), the embedding layer’s output will be a tensor of shape (batch_size, max_length, embedding_dim)'''
        embedded = self.embedding(src)  # shape: (batch_size, max_length, embedding_dim)
        

        '''LSTM will process only the non-padded elements'''
        packed_embedded = pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=False) # 
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        # hidden and cell shapes: (num_layers, batch_size, hidden_dim)
        
        # Note if you need the outputs you need to unpack the sequences using pad_packed_sequence
        # outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True) #outputs shape: (batch_size, max_length, hidden_dim)
    
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
        
        '''Step 3: Generate prediction for the next token'''
        prediction = self.fc_out(output.squeeze(1))  # Shape: (batch_size, output_dim)
        
        return prediction, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,  teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
    def forward(self, src, src_lengths, tgt):
        # src: Source input sequence (English) - Shape: (batch_size, src_sequence_length)
        # src_lengths: Actual lengths of each sequence in src (ignores padding)
        # tgt: Target sequence (French) - Shape: (batch_size, tgt_sequence_length)
        
        batch_size = src.shape[0]
        tgt_sequence_length = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features

        '''Decoder outputs tensor'''
        outputs = torch.zeros(batch_size, tgt_sequence_length, tgt_vocab_size) # output the full prob. distribution over the output vocabulary for each time step in the target sequence, for thecomputing the loss function, but in inference we will only use the argmax of this distribution

        '''Get the hidden and cell states from the encoder'''
        hidden, cell = self.encoder(src, src_lengths)

        '''The first input to the decoder is the <SOS> token for each sequence'''
        input_token = tgt[:, 0]  # Shape: (batch_size)

        '''Go over each token in the target sequence'''
        for t in range(1, tgt_sequence_length):
            '''Input token: is the current token that we’re feeding to the decoder'''
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            
            # Store the output prediction for this time step
            outputs[:, t, :] = output
            
            '''Determine if we should use teacher forcing'''
            teacher_force = random.random() < self.teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else output.argmax(1)

        return outputs
    def inference(self, src, src_lengths, max_len=23, start_token=1, end_token=2):
        # src: Source input sequence - Shape: (batch_size, src_sequence_length) (English)
        # src_lengths: Actual lengths of each sequence in src (ignores padding)
        # max_len: Maximum length of the generated sequence 
        # start_token: Start-of-sequence token index
        # end_token: End-of-sequence token index
        
        batch_size = src.shape[0]

        '''Encode the source sequence'''
        hidden, cell = self.encoder(src, src_lengths)

        '''Initialize the input token with the <SOS> token, for each sequence in the batch'''
        input_token = torch.tensor([start_token] * batch_size) # Shape: (batch_size)

        generated_tokens = torch.zeros(batch_size, max_len).long() # storing the token indices

        for t in range(max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            
            '''Get the most likely token (predicted token) at this time step'''
            input_token = output.argmax(1)  # Shape: (batch_size)
            generated_tokens[:, t] = input_token
            
            '''Stop generating if all sequences in the batch reach <EOS>'''
            if (input_token == end_token).all():
                break
        return generated_tokens
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_src_len, max_tgt_len):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        # Get source and target sentences
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # Convert tokens to indices and pad to max length
        src_indices = self._convert_and_pad(src_sentence, self.src_vocab, self.max_src_len, "<PAD>")
        tgt_indices = self._convert_and_pad(tgt_sentence, self.tgt_vocab, self.max_tgt_len, "<PAD>")

        # Get actual lengths (ignoring <PAD> tokens)
        src_length = min(len(src_sentence), self.max_src_len)
        tgt_length = min(len(tgt_sentence), self.max_tgt_len)

        return torch.tensor(src_indices), src_length, torch.tensor(tgt_indices)
    
    def _convert_and_pad(self, sentence, vocab, max_len, pad_token):
        # Convert tokens to indices
        indices = [vocab.get(token, vocab["<UNK>"]) for token in sentence]
        # Truncate if necessary and pad to max length
        indices = indices[:max_len] + [vocab[pad_token]] * (max_len - len(indices))
        return indices


max_src_len = 25  
max_tgt_len = 25  
batch_size = 32   # Batch size

# Initialize the dataset
train_dataset = TranslationDataset(
    src_sentences=english_tokenized, 
    tgt_sentences=french_tokenized, 
    src_vocab=english_vocab, 
    tgt_vocab=french_vocab, 
    max_src_len=max_src_len, 
    max_tgt_len=max_tgt_len
)
def collate_batch(batch):
    src_batch, src_lengths, tgt_batch = zip(*batch)
    
    # Stack each element of the batch to create the batch tensors
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    src_lengths = torch.tensor(src_lengths)

    return src_batch, src_lengths, tgt_batch

# Initialize the DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_batch(x), num_workers=4)
# Hyperparameters
input_dim = len(english_vocab)
output_dim = len(french_vocab)
embedding_dim = 256
hidden_dim = 512
num_layers = 5
learning_rate = 0.001
teacher_forcing_ratio = 0.5
num_epochs = 20

# Initialize the encoder, decoder, and Seq2Seq model
encoder = Encoder(input_dim, embedding_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, embedding_dim, hidden_dim, num_layers, teacher_forcing_ratio=teacher_forcing_ratio)
model = Seq2Seq(encoder, decoder)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=french_vocab["<PAD>"])


# Initialize an empty list to store the loss for each epoch
epoch_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for src, src_lengths, tgt in train_dataloader:
        optimizer.zero_grad()
        
        # Forward pass through the Seq2Seq model
        output = model(src, src_lengths, tgt)  # Shape: (batch_size, tgt_sequence_length, output_dim)
        
        # Reshape output and target for calculating loss
        output = output[:, 1:].reshape(-1, output_dim)  # Remove the first timestep (<SOS>)
        tgt = tgt[:, 1:].reshape(-1)  # Align target sequence to ignore <SOS>

        # Calculate loss
        loss = criterion(output, tgt)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = epoch_loss / len(train_dataloader)
    epoch_losses.append(avg_loss)
    s = f"seq2seq_model_5_layers_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), s)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

# Save model's state_dict


# Save the losses to a text file
with open('epoch_losses_5.txt', 'w') as f:
    for epoch, loss in enumerate(epoch_losses, 1):
        f.write(f"Epoch {epoch}: Loss = {loss}\n")

