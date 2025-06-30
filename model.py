from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim

# NumPy and Sklearn for data manipulation and evaluation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Optional: tqdm for progress bars
from tqdm import tqdm

from vocab_initializer import generateToken



class MultiPerspectiveNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super().__init__()
    
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True)
  

  def forward(self, inputs, input_size, seq_len, H_C): # this is going to be a trained decoder

    # the forward function here serves to take the last hidden state from the BERT encoder 
    # use this hidden state, and given a correct_text, generate text with until it hits the same length
    # at each iteration or word being generated, use the current hidden state to generate the next predicted word 
    # loss gets evaluated at the end of the entire sentence, and this model only returns the predicted words with the specified length

    if H_C is None: # no hidden state or previous memory to use, new model
        # Initial state with shape: (batch_size, num_hiddens)
        H = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
        C = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
    else:
        H, C = H_C 
        # if there is a hidden state, use the same memory 
    outputs = []
    # outputs is the array of all hidden states
    sentences = []
    for i in range(seq_len):
        # inputs is the last hidden state of the BERT encoder
        # lstm takes in (seq, features) 
        output, (H, C) = self.lstm((seq_len, input_size), (H, C))
        # pass in the hidden and cell state through the LSTM
        predicted_word = generateToken(H)
        sentences.append(predicted_word)
        
        
    return outputs, (H,C), sentences
  

  
  
  