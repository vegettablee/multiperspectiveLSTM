from encoder import initializeModel, bert_model, tokenizer, input_dim, computeLoss
from data_loader import fetchData
import torch

EPOCH_SIZE = 2 
BATCH_SIZE = 10
load_model = False

def load_saved_model(): # place holder data for now 
  model = initializeModel()
  return model

def load_optimizer(): # will change later 
  optimizer = "optimizer"
  return optimizer

def checkpoint(model): 
  # this will save the model at a checkpoint 
  return model


def start_training_loop(): 
  sentences, perspectives, batches = fetchData()
  if load_model == True: 
    model = load_saved_model()
    # load hidden state 
  else : 
    model = initializeModel()
    optimizer = torch.optim.Adam(model.params()) # using adam optimizer 
    H_C=None

  batches = fetchData(BATCH_SIZE)
  num_of_batches = len(batches) 

  for batch in batches: 
    for item in batch:  # item is a tuple in the form of (input, perspectives, outputs) 

      sentence = item[0] # sentence from the data, that gets extracted
      perspectives = item[1]
      correct_outputs = item[2]

      total_loss = 0.0

      inputs = tokenizer(sentence, return_tensors="pt") 
      # using a pre-trained tokenizer because this reduces the stress of creating my own tokenizer and vocabulary 
      
      with torch.no_grad(): # freezes the bert encoding vectors, so the vectors don't change during training 
          encoding = bert_model(**inputs) 
      
      hidden_embeddings = encoding.last_hidden_state # gets the last hidden state at each iteration for the predicted token 
      seq_len = len(sentence)

      for index in range(len(correct_outputs)): # iterating through the separate perspectives/outputs for the same input 
        perspective = perspectives[index]
        correct_output = correct_outputs[index]
        outputs, (H,C), predicted = model(hidden_embeddings, input_dim, seq_len, H_C)
        # we use the hidden states(outputs) at each token, and we need to compute the loss at each predicted token
        H_C = (H,C) 
        # use the updated hidden state for the next sentence 
        loss = computeLoss(outputs, perspective, correct_output, predicted) # get the total for a single sentence plus the tokens
        total_loss += loss # accumulated loss 

  total_loss.backward() 
  optimizer.step()
