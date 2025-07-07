import json 

file_name = "data.json"


def readFile():
  with open(file_name, 'r') as f:
    data = json.load(f)
  return data

def fetchData(batch_size): 
  data = readFile()

  batches = []
  batch = [] 

  counter = 0

  for entry in data: 
    input = entry['input']
    perspectives = entry['Perspectives']
    outputs = entry['Output']
    item = (input, perspectives, outputs) # splits each data into a tuple and appends it to a mini-batch
    print("Batch number : " + str(len(batches) + 1) + "Item added to batch : " + str(item))
    batch.append(item)
    counter += 1

    if batch_size == counter: # mini-batch gets appended to the big batches(epoch) 
      batches.append(batch)
      batch = []
      counter = 0

  return batches

# fetchData(10), for testing