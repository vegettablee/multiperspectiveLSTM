from pathlib import Path
from transformers import AutoTokenizer, BertModel
import torch
import torch.nn as nn

import decoder_model  # after renaming model.py to decoder_model.py

model_dir = "/Users/prestonrank/LSTM_multi_perspective/bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(model_dir)
bert_model = BertModel.from_pretrained(
    model_dir,
    from_tf=False,
    from_flax=False,
    trust_remote_code=False,
    use_safetensors=False  # this is the critical flag
)

input_dim = 768
hidden_size = 512
num_layers = 2

vocab_size = 30522

loss_function = nn.CrossEntropyLoss()

def initializeModel(): # for creating a fresh model to train from scratch
  decoder = decoder_model.MultiPerspectiveNN(input_dim, hidden_size, num_layers)
  return decoder


def train(model, sentences): #
  inputs = tokenizer(sentence, return_tensors="pt") 
# using a pre-trained tokenizer because this reduces the stress of creating my own tokenizer and vocabulary 
  outputs = bert_model(**inputs) 
# running the inputs through the model
  hidden_embeddings = outputs.last_hidden_state 
# gets the last hidden state, which will be used as input for the decoder for text generation
  seq_len = len(sentence)
  outputs, (H,C), sentence = model(hidden_embeddings, input_dim, seq_len, H_C=None)
  return outputs, (H,C), sentence


def saveModel(): # this gets modified later
  PATH = "model_state_dict.pth"
  torch.save(decoder.state_dict(), PATH)

def load_model():
  torch.load


# for word in sentence:
  # print(word)
# hidden states that will be used to evaluate prediction accuracy 