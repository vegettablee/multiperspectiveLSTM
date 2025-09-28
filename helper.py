from pathlib import Path
import torch
import torch.nn as nn
import tokenizer
import decoder_model

loss_function = nn.CrossEntropyLoss()

def computeLoss(model, outputs, correct_ids, attention_contexts):

  losses = []

  for index, (hidden_state, attention_context) in enumerate(zip(outputs, attention_contexts)):
    target_token_id = correct_ids[index]  # get correct token id
    target_tensor = torch.tensor([target_token_id])

    hidden_state = hidden_state[-1, 0, :]  # (hidden_size,)
    attention_context = attention_context.squeeze(0) if attention_context.dim() > 1 else attention_context  # (input_size,)

    # Concatenate hidden state with attention context (same as in generateToken)
    combined_features = torch.cat([hidden_state, attention_context], dim=-1)
    logits = model.linear_layer(combined_features.unsqueeze(0)) # (1, vocab_size)

    loss = loss_function(logits, target_tensor) # compute the loss
    losses.append(loss)

  return sum(losses) 

# seq_len = 10

# model = decoder_model.MultiPerspectiveNN(input_dim, hidden_size, num_layers)

# inputs = tokenizer.tokenizer("The cat ran across the street.", return_tensors="pt") 
      # using a pre-trained tokenizer because this reduces the stress of creating my own tokenizer and vocabulary 

# encoding = tokenizer.bert_model(**inputs)

# cls_token = encoding.last_hidden_state[:, 0, :]
# gets the cls token which holds the input representation of the entire sequence into one vector 

# iterating through the separate perspectives/outputs for the same input 
# outputs, (H,C), predicted = model(cls_token, seq_len, H_C=None)

