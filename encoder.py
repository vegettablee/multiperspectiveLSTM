from pathlib import Path
import torch
import torch.nn as nn
import tokenizer
import decoder_model

model_dir = "/Users/prestonrank/LSTM_multi_perspective/bert-base-uncased"

input_dim = 768
hidden_size = 768
num_layers = 2

vocab_size = 30522

loss_function = nn.CrossEntropyLoss()

def computeLoss(hidden_states, sentence): 
  for hidden_state in hidden_states: 
    print("hello")
  return 5

def initializeModel(): # for creating a fresh model to train from scratch
  decoder = decoder_model.MultiPerspectiveNN(input_dim, hidden_size, num_layers)
  return decoder


def saveModel(): # this gets modified later
  PATH = "model_state_dict.pth"
  torch.save(decoder.state_dict(), PATH)


def load_model():
  torch.load

model = decoder_model.MultiPerspectiveNN(input_dim, hidden_size, num_layers)

inputs = tokenizer.tokenizer("The cat ran across the street.", return_tensors="pt") 
      # using a pre-trained tokenizer because this reduces the stress of creating my own tokenizer and vocabulary 

encoding = tokenizer.bert_model(**inputs)

hidden_embeddings = encoding.last_hidden_state # gets the last hidden state at each iteration for the predicted token 
# iterating through the separate perspectives/outputs for the same input 
outputs, (H,C), predicted = model(hidden_embeddings, input_dim, 10, H_C=None)

