from helper import computeLoss
from data_loader import fetchData
import torch
import tokenizer
import load

from torch.utils.tensorboard import SummaryWriter

# EPOCH_SIZE = 1 
BATCH_SIZE = 10 # 10 items per batch
load_model = False
load_from_checkpoint = True 

def start_training_loop(): 

  if load_model is False: # makes a new model from scratch
    model = load.initializeModel()
    optimizer = load.initializeOptimizer(model, lr=0.0001)  # Lower learning rate for stability
    start_idx = 0
    start_item_idx = 0
  else : # either loads from a checkpoint or loads the optimizer/model(for testing new batches of data) 
    if load_from_checkpoint is True: # resume training 
      model, optimizer, start_idx, start_item_idx = load.load_model_checkpoint()
      C = load.load_lstm_cell_state()
    else : #
      model = load.initializeModel()
      model = load.load_saved_model(model)
      optimizer = load.load_optimizer(model, lr=0.0001)  # Lower learning rate for stability
      C = load.load_lstm_cell_state()
      start_idx = 0
      start_item_idx = 0

  batches = fetchData(BATCH_SIZE) # returns in the form of [batch1, batch2, batch3]
  num_of_batches = len(batches) 

  writer = SummaryWriter(log_dir="runs/exp_loss_tracking") # for tensorboard

  for batch_idx, batch in enumerate(batches): # EPOCH is the current batch index
    if batch_idx < start_idx: # skip until the right index
      continue 
    optimizer.zero_grad() # clear gradients on each batch 

    total_tokens = 0
    correct_tokens = 0
    total_loss = 0.0 # reset accumulated loss
    predicted_token_counts = {}  # Track predicted token frequency
    print("Batch number : " + str(batch_idx))

    for item_idx, item in enumerate(batch):  # item is a tuple in the form of (input, perspectives, outputs) 
      if item_idx < start_item_idx or batch_idx < start_idx:
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
        correct_ids = correct_ids[1:-1]  # Remove CLS and SEP tokens
        # Keep all correct tokens for teacher forcing

        seq_len = len(correct_ids)
        total_tokens += len(correct_ids)

        added_sep_token = f"{sentence} [SEP] {perspective}"
        # added [SEP] token to have the model differentiate the perspective vs the information to extract
        tokenized_input = tokenizer.tokenizer(added_sep_token, return_tensors="pt")

        with torch.no_grad(): # freezes the bert encoding vectors, so the vectors don't change during training
          encoding = tokenizer.bert_model(**tokenized_input)
          cls_token = encoding.last_hidden_state[:, 0, :]

        outputs, (H,C), sentence, predicted_ids, attention_contexts = model(cls_token, correct_ids, H_C=None)
        H_C = (H,C)

        # DEBUG: Track predicted token distribution
        for pred_id in predicted_ids:
          pred_token = tokenizer.tokenizer.decode(pred_id.item())
          predicted_token_counts[pred_token] = predicted_token_counts.get(pred_token, 0) + 1

        common = list(set(correct_ids) & set(predicted_ids)) # get only the shared tokens
        num_correct = len(common)
        correct_tokens += num_correct
        # use the updated hidden state for the next sentence
        loss = computeLoss(model, outputs, correct_ids, attention_contexts) # get the total for a single sentence plus the tokens
        total_loss += loss # accumulated loss 
    
    avg_loss = total_loss.item() / len(batch)
    accuracy = num_correct / total_tokens
    step = batch_idx

    # DEBUG: Print vocabulary distribution
    print(f"\n=== BATCH {batch_idx} SUMMARY ===")
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Predicted token distribution:")
    sorted_tokens = sorted(predicted_token_counts.items(), key=lambda x: x[1], reverse=True)
    for token, count in sorted_tokens[:10]:  # Top 10 most frequent
      percentage = 100 * count / sum(predicted_token_counts.values())
      print(f"  '{token}': {count} ({percentage:.1f}%)")

    # Check if model is stuck on one token
    if len(predicted_token_counts) == 1:
      print("  ⚠️  WARNING: Model is predicting only ONE token!")
    elif len(predicted_token_counts) <= 3:
      print("  ⚠️  WARNING: Very low token diversity!")

    writer.add_scalar("batch/loss", avg_loss, step)
    writer.add_scalar("accuracy/loss", accuracy, step)
  
    total_loss.backward() # backpropagation

    # DEBUG: Monitor gradients BEFORE clipping
    total_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
      if param.grad is not None:
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        param_count += 1
        if 'linear_layer' in name or 'attention' in name:
          print(f"  Grad norm {name}: {param_norm:.6f}")

    total_norm = total_norm ** (1. / 2)
    print(f"  Total gradient norm BEFORE clipping: {total_norm:.6f}")

    # GRADIENT CLIPPING - clip to max norm of 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # DEBUG: Check gradient norm AFTER clipping
    total_norm_after = 0
    for param in model.parameters():
      if param.grad is not None:
        param_norm = param.grad.data.norm(2)
        total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** (1. / 2)
    print(f"  Total gradient norm AFTER clipping: {total_norm_after:.6f}")

    optimizer.step() # gradient descent 

    load.save_checkpoint(model, optimizer, batch_idx, item_idx)# save model at each batch/epoch as a checkpoint 
    load.save_lstm_cell_state(C)

  writer.flush()
  writer.close()  
  # tensorboard --logdir runs/exp_loss_tracking, run this command to 

start_training_loop()
