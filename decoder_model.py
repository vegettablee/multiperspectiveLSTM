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
    self.linear_layer = nn.Linear(hidden_size + input_size, vocab_size)  # concatenate attention output

    bert_embeddings = tokenizer.bert_model.embeddings.word_embeddings

    embedding_dim = 768

    self.embedding_layer = nn.Embedding(
    num_embeddings=bert_embeddings.num_embeddings,
    embedding_dim=bert_embeddings.embedding_dim
    )

    self.embedding_layer.weight.data = bert_embeddings.weight.data.clone()

    self.embedding_layer.weight.requires_grad = True   # set false to stop training

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True)

    # Attention mechanism components
    self.attention_query = nn.Linear(hidden_size, hidden_size)
    self.attention_key = nn.Linear(input_size, hidden_size)
    self.attention_value = nn.Linear(input_size, input_size)

    # Initialize attention weights properly
    nn.init.xavier_uniform_(self.attention_query.weight)
    nn.init.zeros_(self.attention_query.bias)
    nn.init.xavier_uniform_(self.attention_key.weight)
    nn.init.zeros_(self.attention_key.bias)
    nn.init.xavier_uniform_(self.attention_value.weight)
    nn.init.zeros_(self.attention_value.bias)

    # Initialize linear layer with smaller weights
    nn.init.xavier_uniform_(self.linear_layer.weight, gain=0.1)  # Smaller gain for stability
    nn.init.zeros_(self.linear_layer.bias)

  def attention(self, query, keys, values):
    # query: (batch_size, hidden_size) - current LSTM hidden state
    # keys: (batch_size, seq_len, input_size) - all previous token embeddings
    # values: (batch_size, seq_len, input_size) - all previous token embeddings

    if keys.size(1) == 0:  # no previous tokens
      return torch.zeros(query.size(0), self.input_size, device=query.device)

    # Transform query and keys to same dimension
    query_transformed = self.attention_query(query).unsqueeze(1)  # (batch_size, 1, hidden_size)
    keys_transformed = self.attention_key(keys)  # (batch_size, seq_len, hidden_size)
    values_transformed = self.attention_value(values)  # (batch_size, seq_len, input_size)

    # Compute attention scores
    scores = torch.bmm(query_transformed, keys_transformed.transpose(1, 2))  # (batch_size, 1, seq_len)
    scores = scores / (self.hidden_size ** 0.5)  # scale

    # Apply softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, 1, seq_len)

    # DEBUG: Check attention distribution
    if keys.size(1) > 1:  # Only if we have multiple tokens to attend to
      attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
      max_attention = torch.max(attention_weights, dim=-1)[0]
      print(f"  Attention entropy: {attention_entropy.item():.4f} (higher=more diverse)")
      print(f"  Max attention weight: {max_attention.item():.4f} (lower=more diverse)")
      print(f"  Attention weights: {attention_weights.squeeze().detach().numpy()}")

    # Apply attention weights to values
    context = torch.bmm(attention_weights, values_transformed)  # (batch_size, 1, input_size)

    return context.squeeze(1)  # (batch_size, input_size)

  def generateToken(self, hidden_state, attention_context):

    if hidden_state.dim() == 3:
            # hidden_state shape: (num_layers, batch_size, hidden_size)
            # Take the last layer
            hidden_state = hidden_state[-1, :, :]  # (batch_size, hidden_size)

    # Concatenate hidden state with attention context
    combined_features = torch.cat([hidden_state, attention_context], dim=-1)
    logits = self.linear_layer(combined_features)

    # DEBUG: Check logits distribution
    logits_std = torch.std(logits)
    logits_max = torch.max(logits)
    logits_min = torch.min(logits)

    token_probs = nn.Softmax(dim=-1)(logits)

    # DEBUG: Check probability distribution
    prob_entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-8))
    max_prob = torch.max(token_probs)
    top_5_probs, top_5_indices = torch.topk(token_probs, 5)

    index = torch.argmax(token_probs, dim=-1)
    predicted_word = tokenizer.tokenizer.decode(index.item())

    print(f"  Logits stats: std={logits_std:.4f}, range=[{logits_min:.2f}, {logits_max:.2f}]")
    print(f"  Prob entropy: {prob_entropy.item():.4f} (higher=more diverse)")
    print(f"  Max probability: {max_prob.item():.4f}")
    print(f"  Top 5 probs: {top_5_probs.detach().numpy()}")
    print(f"  Generated token: '{predicted_word}' (ID: {index.item()})")

    return predicted_word, index

  def projectVector(self, v1, v2): # v2 has shape (2, 768) 
     projV = torch.zeros(2,768)
     dims = v2[1].shape # returns 2
     v1 = torch.flatten(v1) # turns into 1 tensor (768) 
     for dim in range(2): 
        v3 = v2[dim, :]
        dot_product = torch.dot(v1, v3)
        magnitude = torch.dot(v3, v3)
        if magnitude == 0: 
          projV[dim, :] = torch.zeros_like(v1)
        else:
          scalar_factor = dot_product / magnitude
          projected_vector = scalar_factor * v3
          projV[dim, :] = projected_vector

     return projV # shape (2, 768)
  
  def forward(self, cls_token, correct_ids, H_C=None): # teacher forcing with correct tokens

    # Teacher forcing: always feed the correct previous tokens as input
    # Use attention over all previous correct tokens to predict the next token
    # This helps the model learn faster and more accurately

    batch_size = cls_token.size(0) # this is always 1
    seq_len = len(correct_ids)
    sentences = []

    if H_C==None: # no previous hidden memory
      H = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=cls_token.device)
      C = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=cls_token.device)
    else :
      (H,C) = H_C

    v2 = torch.flatten(H, start_dim=0, end_dim=1) # this turns into shape (2, 768), cls token projects onto each separate vector

    H = self.projectVector(cls_token, v2) # shape (2, 768)
    H = H.unsqueeze(1) # turns into shape (2, 1 768)

    outputs = []
    predicted_ids = []
    attention_contexts = []  # Store attention contexts for loss computation

    # Keep track of all previous correct token embeddings for attention
    previous_embeddings = []  # Will store embeddings of previous correct tokens

    for i in range(seq_len):
        # Get current correct token
        current_token_id = correct_ids[i]
        current_word = tokenizer.tokenizer.decode(current_token_id)

        # Convert to embedding
        converted_token = torch.tensor([current_token_id], device=cls_token.device)
        current_embedding = self.embedding_layer(converted_token) # shape (1, 768)
        current_embedding = current_embedding.unsqueeze(0) # shape (1, 1, 768)

        # Pass through LSTM
        output, (H, C) = self.lstm(current_embedding, (H, C))

        # DEBUG: Check hidden state saturation
        h_mean = torch.mean(torch.abs(H))
        h_max = torch.max(torch.abs(H))
        h_std = torch.std(H)
        c_mean = torch.mean(torch.abs(C))

        # Check for saturation (values close to -1 or 1 for tanh)
        h_saturated = torch.sum(torch.abs(H) > 0.9).item()
        h_total = H.numel()

        if i == 0 or i % 5 == 0:  # Print every 5 steps to avoid spam
          print(f"  Hidden state stats - mean: {h_mean:.4f}, max: {h_max:.4f}, std: {h_std:.4f}")
          print(f"  Cell state mean: {c_mean:.4f}")
          print(f"  Saturated units: {h_saturated}/{h_total} ({100*h_saturated/h_total:.1f}%)")

        # Compute attention over all previous token embeddings
        if previous_embeddings:
            # Stack previous embeddings: (batch_size, seq_len, input_size)
            keys_values = torch.stack(previous_embeddings, dim=1)  # (1, i, 768)
            attention_context = self.attention(H[-1, :, :], keys_values, keys_values)
        else:
            attention_context = torch.zeros(batch_size, self.input_size, device=cls_token.device)

        # Generate prediction using attention context
        predicted_word, predicted_id = self.generateToken(H, attention_context)

        predicted_ids.append(predicted_id)
        sentences.append(current_word)  # Use correct word for teacher forcing
        outputs.append(H)
        attention_contexts.append(attention_context)  # Store for loss computation

        # Add current embedding to previous embeddings for next iteration
        previous_embeddings.append(current_embedding.squeeze(0))  # Remove batch dim: (1, 768)

        print(f"Step {i}: Correct='{current_word}', Predicted='{predicted_word}'")

    return outputs, (H,C), sentences, predicted_ids, attention_contexts
  

  # there are two main approaches I want to try, this is for remembering later, the first one is :
  # the first one is intializing the LSTM with the hidden state as the BERT's contextual last hidden state, and having the model
  # keep using this hidden state and letting it evolve through each token generated 

  # the second is :
  # instead of initializing the LSTM with the hidden state, we add the bert's hidden state to the input of the LSTM, resulting in 
  # an input with a higher dimension, but the model can choose what parts of the input features to actually use, which would be 
  # a possible approach as well 

  # some more approaches would be the append the CLS token to each embedding input, to try to remember context 
  # another approach would be to add an attention over bert's outputs, so it can learn what needs to be extracted
  # and using teacher forcing, instead of passing in the previous generated token at each iteration, pass in the correct token 
  # and have it generate, so it can learn the correct token quicker 

  # maybe instead of reinitializing the hidden state with bert's last hidden state each time
  # use the hidden state 