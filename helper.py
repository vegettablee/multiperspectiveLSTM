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

def computeLoss(model, outputs, correct_ids):

  losses = []
  
  for index, hidden_state in enumerate(outputs): 
    target_token_id = correct_ids[index]  # get correct token id 
    target_tensor = torch.tensor([target_token_id])

    hidden_state = hidden_state[-1, 0, :]
    logits = model.linear_layer(hidden_state.unsqueeze(0)) # turns shape into (1, 768)
    # hidden_state has shape (N,L, hidden_out), needs to be transformed into (1, hidden_out)

    loss = loss_function(logits, target_tensor) # compute the loss 
    losses.append(loss) 

  return sum(losses) 

def initializeModel(): # for creating a fresh model to train from scratch
  decoder = decoder_model.MultiPerspectiveNN(input_dim, hidden_size, num_layers)
  return decoder


def saveModel(): # this gets modified later
  PATH = "model_state_dict.pth"
  torch.save(decoder.state_dict(), PATH)


def load_model():
  torch.load

# seq_len = 10

# model = decoder_model.MultiPerspectiveNN(input_dim, hidden_size, num_layers)

# inputs = tokenizer.tokenizer("The cat ran across the street.", return_tensors="pt") 
      # using a pre-trained tokenizer because this reduces the stress of creating my own tokenizer and vocabulary 

# encoding = tokenizer.bert_model(**inputs)

# cls_token = encoding.last_hidden_state[:, 0, :]
# gets the cls token which holds the input representation of the entire sequence into one vector 

# iterating through the separate perspectives/outputs for the same input 
# outputs, (H,C), predicted = model(cls_token, seq_len, H_C=None)

