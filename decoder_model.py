from sentence_transformers import SentenceTransformer
import tokenizer
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
    
    bert_embeddings = tokenizer.bert_model.embeddings.word_embeddings

    embedding_dim = 768

    self.embedding_layer = nn.Embedding(
    num_embeddings=bert_embeddings.num_embeddings,
    embedding_dim=bert_embeddings.embedding_dim
    ) 

    self.embedding_layer.weight.data = bert_embeddings.weight.data.clone()

    self.embedding_layer.weight.requires_grad = True   # set false to stop training

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True)
  
  def generateToken(self, hidden_state): 

    if hidden_state.dim() == 3:
            # hidden_state shape: (num_layers, batch_size, hidden_size)
            # Take the last layer
            hidden_state = hidden_state[-1, :, :]  # (batch_size, hidden_size)
    
    logits = self.linear_layer(hidden_state)

    token_probs = nn.Softmax(dim=-1)(logits)
    index = torch.argmax(token_probs, dim=-1) 

    predicted_word = tokenizer.tokenizer.decode(index.item())
    print("generated token : " + predicted_word)
    return predicted_word, index
  
  def forward(self, cls_token, seq_len, first_token, H_C=None): # this is going to be a trained decoder

    # the forward function here serves to take the last hidden state from the BERT encoder, specifically the CLS token
    # use this hidden state, and given a correct_text, generate text with until it hits the same length
    # at each iteration or word being generated, use the current hidden state to generate the next predicted word 
    # loss gets evaluated at the end of the entire sentence, and this model only returns the predicted words with the specified length

    batch_size = cls_token.size(0) # this is always 1
    sentences = []
    
    if H_C==None: # no previous hidden memory 
      H = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=cls_token.device)
      C = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=cls_token.device)
    else : 
      (H,C) = H_C
      # H is a tensor with shape (D * num_layers, batch_size, hidden_size)

    inputs = cls_token.unsqueeze(0).repeat(2, 1, 1)  # Shape: (2, 1, 768)
    # unsqueeze adds a new dimension at position 0 
    # repeat, repeats the dimension of the first index twice, and repeats the other dimensions once

    H = inputs 
    # initialize the LSTM with BERT's hidden state, the cls_token

    outputs = []
    # outputs is the array of all hidden states throughout
    
    first_word = tokenizer.tokenizer.decode(first_token)
    converted_token = torch.tensor([first_token])
    inputs = self.embedding_layer(converted_token) # one dimensional vector of size 768, shape (1, 768)
    inputs = inputs.unsqueeze(0) # to fix shape into (1, 1, 768)
    sentences.append(first_word)
    print("First input token: " + first_word) 
    predicted_ids = []
    for i in range(seq_len):
        # lstm takes in (seq, features) 
        output, (H, C) = self.lstm(inputs, (H, C))
        # pass in the hidden and cell state through the LSTM recursively
        predicted_word, index = self.generateToken(H)
        predicted_ids.append(index)
        # get predicted_word 
        input_id = tokenizer.tokenizer(predicted_word, return_tensors="pt")['input_ids'] # tokenizer returns shape (1, 1)
        # tokenize the predicted word
        inputs = self.embedding_layer(input_id)
        sentences.append(predicted_word)    
        outputs.append(H)
        
    return outputs, (H,C), sentences, predicted_ids
  

  # there are two main approaches I want to try, this is for remembering later, the first one is :
  # the first one is intializing the LSTM with the hidden state as the BERT's contextual last hidden state, and having the model
  # keep using this hidden state and letting it evolve through each token generated 

  # the second is :
  # instead of initializing the LSTM with the hidden state, we add the bert's hidden state to the input of the LSTM, resulting in 
  # an input with a higher dimension, but the model can choose what parts of the input features to actually use, which would be 
  # a possible approach as well 