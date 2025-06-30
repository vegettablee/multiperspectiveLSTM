from transformers import AutoTokenizer, TFBertModel
import torch
import torch.nn as nn

from model import MultiPerspectiveNN

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = TFBertModel.from_pretrained("google-bert/bert-base-uncased")

input_dim = 768
hidden_size = 512
sigma = 0.01
num_layers = 2

vocab_size = 30522

loss_function = nn.crossEntropyLoss()
# compares the correct token with the predicted token

linear_layer = nn.Linear(input_dim, vocab_size) 
# turns the raw hidden state into a logits for each token, hence the size of the input(BERT embeddings) and the output(vocab size)

lstm = nn.LSTM(input_dim, hidden_size, num_layers, bias=True)
# LSTM used to generate text based on the output length inside of the training data

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt") 
# using a pre-trained tokenizer because this reduces the stress of creating my own tokenizer and vocabulary 

outputs = model(inputs) 
# running the inputs through the model

hidden_embeddings = outputs.last_hidden_state 
# gets the last hidden state, which will be used as input for the decoder for text generation

hidden_states = decoder(hidden_embeddings)
# hidden states that will be used to evaluate prediction accuracy 

def generateToken(hidden_state): 
  layer = linear_layer(hidden_state)
  predicted_word = nn.softMax(layer)
  return predicted_word