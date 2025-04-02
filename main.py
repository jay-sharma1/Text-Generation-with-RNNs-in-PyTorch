import sys
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim

STANDARD_SEQUENCE_LENGTH = 50
HIDDEN_DIM_SIZE = 256
NUM_LAYERS = 1
EPOCHS = 30
LR = 0.015

'''
STEP 1: Prepare the Dataset
'''
def prepareText(filename: str, seq_length = STANDARD_SEQUENCE_LENGTH):
    # Tokenize the text into lowercase characters .
    tokens = None
    
    with open(filename, "r") as f:
        tokens = list(f.read().lower())

    # Convert tokens into numerical indices (word2idx mapping).
    token_counts = Counter(tokens)
    
    # vocab is a list containing each unique word within the input text, sorted by its frequency
    vocab = sorted(token_counts, key=token_counts.get, reverse=True)
    
    # A dictionary of indexes and their corresponding token
    ind_to_token = {i: ch for i, ch in enumerate(vocab)}
    
    # A dictionary of tokens and their corresponding index
    token_to_ind = {ch: i for i, ch in enumerate(vocab)}
    
    
    print("The number of characters in the text is: ", len(tokens))
    print("The number of unique characters is:", len(vocab))
    
    sequences = []
    targets = []
    
    # Sliding window approach where we take every group of 50 characters and add their corresponding index to
    # sequences (e.g., [h, e, l, l, o, ' ', w, o, r, l, d] -> [0, 11, 23, 3, 5, 99]), while targets stores
    # the index of the start of the next window for use during training. When the sequence is used for training,
    # the predicted value is compared to the corresponding index in targets to see if it predicted the next character correctly.
    for i in range(0, len(tokens) - seq_length):
        sequences.append([token_to_ind[ch] for ch in tokens[i:i+seq_length]])
        targets.append(token_to_ind[tokens[i+seq_length]])
    
    print("Total Patterns: ", len(sequences))
    
    # Convert the arrays into tensors, which are a data structure used to hold a multi-dimensional
    # matrix that consists of a single data-type. They are similar to numpy arrays, except tensors support
    # gpu computing (more relevant for deep learning models) and autograd (tracks the history of every computation
    # and thus speeds up the computation of the loss function). 
    
    # Reshape sequences to be [samples, time steps, features]
    sequences = torch.tensor(sequences, dtype=torch.float32).reshape(len(sequences), seq_length, 1)
    sequences = sequences / float(len(vocab))
    targets = torch.tensor(targets)
    print(sequences.shape, targets.shape)
    
    return len(vocab)

'''
STEP 2: Define the RNN Model

vocab_size: Size of your vocabulary (number of unique tokens).

embedding_dim: The size of each embedding vector (i.e., the size of the vector that each token is embedded into).

hidden_dim: Number of features within the hidden state in the LSTM.

n_layers: Number of LSTM layers (e.g., 2 or 3 for deeper networks).
'''
class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

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
        
        input_size: The input is a single feature (i.e. one integer for a single character)
        hidden_size: Number of features within the hidden state in the LSTM.
        num_layers: Number of LSTM layers (e.g., 2 or 3 for deeper networks).
        batch_first: Input and output tensors are provided as (batch, seq, feature)
        '''
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN_DIM_SIZE, num_layers=NUM_LAYERS, batch_first=True)
        
        # During training, randomly zeroes some of the elements of the input tensor with probability p. It is
        # proven to be an effective technique for regularization.
        self.dropout = nn.Dropout(0.2)
        
        # Applies an affine linear transformation to the incoming data. The fully connected layer for output predictions.
        self.fc = nn.Linear(HIDDEN_DIM_SIZE, vocab_size)
    

    def forward(self, x):
        # Take only the last output, which is the feature.
        x, _ = self.lstm(x)
        
        # Produce output
        x = x[:, -1, :]
       
        x = self.linear(self.dropout(x))
        return x
    
'''
STEP 3: Train the Model
'''

def trainModel():
    pass

if __name__ == "__main__":
    filename = sys.argv[1]
    
    vocab_size = prepareText(filename)
    
    model = TextGenerationModel(vocab_size=vocab_size)
    