from collections import Counter
import numpy as np
import torch
from torch import nn as nn

STANDARD_SEQUENCE_LENGTH = 50
FILE_NAME = "shakespeare.txt"
EMBEDDING_DIM_SIZE = 128
HIDDEN_DIM_SIZE = 256
NUM_LAYERS = 2
EPOCHS = 30
LR = 0.015

# Tokenize and preprocess the text
def tokenize_text(text: str) -> tuple[list[str], dict, dict]:
    # Convert text to lowercase and split into characters or words
    tokens = list(text.lower()) 
    
    # Creatres a frequency dictionary of all words within the input text
    token_counts = Counter(tokens)
    
    # vocab is a list containing each unique word within the input text, sorted by its frequency
    vocab = sorted(token_counts, key=token_counts.get, reverse=True)
    
    # A dictionary of indexes and their corresponding token
    int_to_token = {i: ch for i, ch in enumerate(vocab)}
    
    # A dictionary of tokens and their corresponding index
    token_to_int = {ch: i for i, ch in enumerate(vocab)}
    
    return tokens, token_to_int, int_to_token

# Text is split into fixed-length sequences. Standard length is set within STANDARD_SEQUENCE_LENGTH,
# or a custom length can be taken as input.
def create_sequences(tokens: list[str], token_to_int: dict, seq_length = STANDARD_SEQUENCE_LENGTH) -> tuple[np.array, np.array]:
    sequences = []
    targets = []
    
    # Sliding window approach where we take every group of 50 words and add their corresponding index to
    # sequences (e.g., [hello, world, my, name, is, blank] -> [0, 11, 23, 3, 5, 99]), while targets stores
    # the index of the start of the next window.
    for i in range(0, len(tokens) - seq_length):
        sequences.append([token_to_int[ch] for ch in tokens[i:i+seq_length]])
        targets.append(token_to_int[tokens[i+seq_length]])
        
    return np.array(sequences), np.array(targets)

'''
Actually creating the RNN model using pytorch.

vocab_size: Size of your vocabulary (number of unique tokens).

embedding_dim: The size of each embedding vector (i.e., the size of the vector that each token is embedded into).

hidden_dim: Number of features within the hidden state in the LSTM.

n_layers: Number of LSTM layers (e.g., 2 or 3 for deeper networks).
'''
class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, n_layers: int):
        super(TextGenerationModel, self).__init__()
        
        # The embedding layer within a NN transforms tokens into vectors of a fixed size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        '''
        RNNs face challenges in learning long-term dependencies where information from distant time steps becomes 
        crucial for making accurate predictions for current state. This is known as the vanishing gradient problem.
        
        During training, the loss function (C) indicates how well the model predicted the the output.
        The network then performs back propagation to minimize the loss function by modifying the weights and biases
        of the network. Updating these values depends on the activation function, which in this case is the sigmoid
        function.The new weight of a node is highly dependent on the gradient of the loss function, which is in turn dependent
        on the partial derivative of the activaiton function. With a sigmoid activation function, the partial derivative
        continues to get smaller for each layer it back propagates, resulting in a loss function that approaches zero.
        
        The solution to this is to use an LSTM (Long Short-Term Memory), which ntroduce a memory cell that holds information 
        over extended periods addressing the challenge of learning long-term dependencies.
        '''
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        
        # Applies an affine linear transformation to the incoming data. The fully connected layer for output predictions
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    '''
    x: A sequence, which is an array of indices generated in the preprocessing step
    hidden: 
    '''
    def forward(self, x, hidden):
        # Converts input token indices into dense vector embeddings.
        x = self.embedding(x)
        
        # The embeddings are passed through an LSTM
        out, hidden = self.lstm(x, hidden)
        
        # Maps the LSTM's hidden states to the vocabulary space, predicting the next token at each time step.
        out = self.fc(out[:, -1, :]) 
        
        return out, hidden


def train_model(model, data_loader, epochs=EPOCHS, lr=LR):
    # The loss function that measures the difference between the predicted output and actual output.
    criterion = nn.CrossEntropyLoss()
    
    # The optimizer function used to minimize the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
    for epoch in range(epochs):
        hidden = None
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, hidden = model(inputs, hidden)
            hidden = tuple([h.detach() for h in hidden])  # Detach hidden state to prevent memory leaks
            
            # Compute loss and backpropagate
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


def generate_text(model, seed_text, token_to_int, int_to_token, length=100):
    model.eval()
    generated = [token_to_int[ch] for ch in seed_text.lower()]
    hidden = None
    
    for _ in range(length):
        inputs = torch.tensor(generated[-50:], dtype=torch.long).unsqueeze(0)
        output, hidden = model(inputs, hidden)
        pred_token = torch.argmax(output, dim=1).item()
        generated.append(pred_token)
    
    return ''.join([int_to_token[idx] for idx in generated])


if __name__ == "__main__":
    tokens, token_to_int, int_to_token = None, None, None
    
    with open(FILE_NAME, "r") as f:
        tokens, token_to_int, int_to_token = tokenize_text(f.read())
        
    sequences, targets = create_sequences(tokens, token_to_int)
    print(tokens)
    
    RNNmodel = TextGenerationModel(len(token_to_int), EMBEDDING_DIM_SIZE, HIDDEN_DIM_SIZE, NUM_LAYERS)
    train_model(RNNmodel, data_loader)
    generate_text(RNNmodel, input("Input the starting text you would like to use: "), token_to_int, int_to_token)