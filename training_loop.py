from encoder import initializeModel, train
from data_loader import fetchData

EPOCH_SIZE = 2 
load_model = False

def start_training_loop(): 
  sentences, perspectives, batches = fetchData()
  if load_model == True: 
    model = load_saved_model
  else : 
    model = initializeModel()

  for EPOCH in EPOCH_SIZE:
    train(model, )
    


def load_saved_model(): 
  model = "this is saved model placeholder"
  return model