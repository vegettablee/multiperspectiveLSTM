from helper import computeLoss
from data_loader import fetchData
import torch
import tokenizer
import load

from torch.utils.tensorboard import SummaryWriter

# EPOCH_SIZE = 1 
BATCH_SIZE = 10 # 10 items per batch
load_model = True
load_from_checkpoint = True # if this is false, run time is 

def start_training_loop(): 

  if load_model is False: # makes a new model from scratch
    model = load.initializeModel()
    optimizer = load.initializeOptimizer(model)
    start_idx = 0
    start_item_idx = 0
  else : # either loads from a checkpoint or loads the optimizer/model(for testing new batches of data) 
    if load_from_checkpoint is True: # resume training 
      model, optimizer, start_idx, start_item_idx = load.load_model_checkpoint()
      C = load.load_lstm_cell_state()
    else : # 
      model = load.initializeModel()
      model = load.load_saved_model(model)
      optimizer = load.load_optimizer(model) 
      C = load.load_lstm_cell_state() 
      start_idx = 0
      start_item_idx = 0

  batches = fetchData(BATCH_SIZE) # returns in the form of [batch1, batch2, batch3]
  num_of_batches = len(batches) 

  writer = SummaryWriter(log_dir="runs/exp_loss_tracking") # for tensorboard

  for batch_idx, batch in enumerate(batches): # EPOCH is the current batch index
    if batch_idx < start_idx: # skip until the right index
      continue 
    total_tokens = 0 
    correct_tokens = 0
    optimizer.zero_grad() # cldear gradients
    total_loss = 0.0 # reset accumulated loss 
    print("Batch number : " + str(batch_idx))

    for item_idx, item in enumerate(batch):  # item is a tuple in the form of (input, perspectives, outputs) 
      if item_idx < start_item_idx:
        continue

      print(item)
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
        total_tokens += len(correct_ids)

        # so the model can generate the correct number of tokens, and this can be used directly for the loss function
        # subtracting one because i manually generate one token before the forward function 

        added_sep_token = f"{sentence} [SEP] {perspective}"
        # added [SEP] token to have the model differentiate the perspective vs the information to extract
        tokenized_input = tokenizer.tokenizer(added_sep_token, return_tensors="pt")

        with torch.no_grad(): # freezes the bert encoding vectors, so the vectors don't change during training 
          encoding = tokenizer.bert_model(**tokenized_input) 
          cls_token = encoding.last_hidden_state[:, 0, :] 

        outputs, (H,C), sentence, predicted_ids = model(cls_token, seq_len, first_token, H_C=None)

        common = list(set(correct_ids) & set(predicted_ids)) # get only the shared tokens
        num_correct = len(common) 
        correct_tokens += num_correct 
        # use the updated hidden state for the next sentence 
        loss = computeLoss(model, outputs, correct_ids) # get the total for a single sentence plus the tokens
        total_loss += loss # accumulated loss 
    
    avg_loss = total_loss.item() / len(batch)
    accuracy = num_correct / total_tokens 
    step = batch_idx
    writer.add_scalar("batch/loss", avg_loss, step)
    writer.add_scalar("accuracy/loss", accuracy, step)
  
    total_loss.backward() # compute 
    optimizer.step()

    load.save_checkpoint(model, optimizer, batch_idx, item_idx)# save model at each batch/epoch as a checkpoint 
    load.save_lstm_cell_state(C)

  writer.flush()
  writer.close()  
  # tensorboard --logdir runs/exp_loss_tracking, run this command to 

start_training_loop()
