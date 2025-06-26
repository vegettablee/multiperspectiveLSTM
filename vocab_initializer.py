from transformers import AutoTokenizer, BertForPreTraining
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForPreTraining.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt") 
# using a pre-trained tokenizer because this reduces the stress of creating my own tokenizer and vocabulary 
outputs = model(**inputs) 
# the outputs a

prediction_logits = outputs.prediction_logits # this will be the explanation with the correct logits used for training 
seq_relationship_logits = outputs.seq_relationship_logits