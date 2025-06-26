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
  def __init__(self, input_dim, hidden_dims, output_dim):
    super().__init__()

    layers = [] # array for each layer
    current_dim = input_dim

    for h_dim in hidden_dims: # simple appends each layer based on the dimensions 
      layers.append(nn.Linear(current_dim, h_dim)) 
      layers.append(nn.ReLU())
      current_dim = h_dim

    # layers.append(nn.Linear(current_dim,output_dim))
    # layers.append(nn.Sigmoid()) 

    init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
    triple = lambda: (init_weight(num_inputs, num_hiddens),
                      init_weight(num_hiddens, num_hiddens),
                      nn.Parameter(torch.zeros(num_hiddens)))
    
    
    self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
    self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
    self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
    self.W_xc, self.W_hc, self.b_c = triple()  # Input node

    self.model = nn.Sequential(*layers)


  def forward(self, inputs, H_C): # this is going to be a train decoder
    if H_C is None: # no hidden state or previous memory to use, new model
        # Initial state with shape: (batch_size, num_hiddens)
        H = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
        C = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
    else:
        H, C = H_C # if there is a hidden state, use the same memory 
    outputs = []

    for X in inputs: # inputs is a token array, each token is passed through each gate
        I = torch.sigmoid(torch.matmul(X, self.W_xi) +
                        torch.matmul(H, self.W_hi) + self.b_i)
        F = torch.sigmoid(torch.matmul(X, self.W_xf) +
                        torch.matmul(H, self.W_hf) + self.b_f)
        O = torch.sigmoid(torch.matmul(X, self.W_xo) +
                        torch.matmul(H, self.W_ho) + self.b_o)
        C_tilde = torch.tanh(torch.matmul(X, self.W_xc) +
                           torch.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * torch.tanh(C)
        outputs.append(H)


    return outputs, (H,C)
  

  
  
  