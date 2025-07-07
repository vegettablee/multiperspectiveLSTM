from helper import initializeModel, input_dim, computeLoss
from data_loader import fetchData
import torch
import tokenizer

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

  if load_model == True: 
    model = load_saved_model()
    # load hidden state 
  else : 
    model = initializeModel()
    optimizer = torch.optim.Adam(model.parameters()) # using adam optimizer 
    H_C=None

  batches = fetchData(BATCH_SIZE)
  num_of_batches = len(batches) 

  for batch in batches: 

    optimizer.zero_grad()
    total_loss = 0.0

    for item in batch:  # item is a tuple in the form of (input, perspectives, outputs) 
      
      sentence = item[0] # sentence from the data, that gets extracted
      perspectives = item[1]
      correct_outputs = item[2]

      for index in range(len(correct_outputs)): # iterating through the separate perspectives/outputs for the same input 
        
        perspective = perspectives[index]
        correct_output = correct_outputs[index]
        
        correct_ids = tokenizer.tokenizer(correct_output).input_ids 
        # get only the input ids of the correct output 
        first_token = correct_ids[1]
        # save the first real token, index 0 is the CLS token
        correct_ids = correct_ids[2:] 
        # slice off the second element, as in the model, it takes the generates based off the first correct token

        seq_len = len(correct_ids)
        # so the model can generate the correct number of tokens, and this can be used directly for the loss function
        # subtracting one because i manually generate one token before the forward function 

        added_sep_token = sentence + " [SEP]" + perspective
        # added [SEP] token to have the model differentiate the perspective vs the information to extract
        tokenized_input = tokenizer.tokenizer(added_sep_token, return_tensors="pt")

        with torch.no_grad(): # freezes the bert encoding vectors, so the vectors don't change during training 
          encoding = tokenizer.bert_model(**tokenized_input) 
          cls_token = encoding.last_hidden_state[:, 0, :] 

        outputs, (H,C), predicted = model(cls_token, seq_len, first_token, H_C=None)

        # use the updated hidden state for the next sentence 
        loss = computeLoss(model, outputs, correct_ids) # get the total for a single sentence plus the tokens
        total_loss += loss # accumulated loss 
  
  total_loss.backward() 
  optimizer.step()

start_training_loop()