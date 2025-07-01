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




class MultiPerspectiveNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    vocab_size = 30522
    self.linear_layer = nn.Linear(hidden_size, vocab_size)

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True)
  
  def generateToken(self, hidden_state): 
    layer = self.linear_layer(hidden_state)
    softmax = nn.Softmax(dim=-1)
    predicted_word = softmax(layer)
    return predicted_word
  
  def forward(self, inputs, input_size, seq_len, H_C): # this is going to be a trained decoder

    # the forward function here serves to take the last hidden state from the BERT encoder 
    # use this hidden state, and given a correct_text, generate text with until it hits the same length
    # at each iteration or word being generated, use the current hidden state to generate the next predicted word 
    # loss gets evaluated at the end of the entire sentence, and this model only returns the predicted words with the specified length

  
    batch_size = inputs.size(0) # this is always 1
    H = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=inputs.device)
    C = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=inputs.device)

    outputs = []
    # outputs is the array of all hidden states
    sentences = []
    for i in range(seq_len):
        # inputs is the last hidden state of the BERT encoder
        # lstm takes in (seq, features) 
        output, (H, C) = self.lstm(inputs, (H, C))
        # pass in the hidden and cell state through the LSTM
        predicted_word = self.generateToken(H)
        sentences.append(predicted_word)    
    return outputs, (H,C), sentences
  

  
  
  