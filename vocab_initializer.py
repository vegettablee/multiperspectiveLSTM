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

decoder = decoder_model.MultiPerspectiveNN(input_dim, hidden_size, num_layers)

loss_function = nn.CrossEntropyLoss()
# compares the correct token with the predicted token
# turns the raw hidden state into a logits for each token, hence the size of the input(BERT embeddings) and the output(vocab size)

lstm = nn.LSTM(input_dim, hidden_size, num_layers, bias=True)
# LSTM used to generate text based on the output length inside of the training data

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt") 
# using a pre-trained tokenizer because this reduces the stress of creating my own tokenizer and vocabulary 

outputs = bert_model(**inputs) 
# running the inputs through the model

hidden_embeddings = outputs.last_hidden_state 
# gets the last hidden state, which will be used as input for the decoder for text generation

outputs, (H,C), sentence = decoder(hidden_embeddings, input_dim, 10, H_C=None)

for word in sentence:
  print(word)
# hidden states that will be used to evaluate prediction accuracy 