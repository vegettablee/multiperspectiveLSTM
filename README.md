To run this code, since I had issues regarding importing the BERT model, I downloaded the model and imported it. Everything needed is inside of the tokenizer.py file.

At a high level, this code is taking a sentence like "The cat ate the pizza on the table." And then, given a perspective like "Cat", generate a sentence based on the perspective, something like 
"The pizza was delicious."

Sentence -> Decoding into BERT Tokens/Embeddings -> Extracting [CLS] token as input to LSTM's hidden state -> Based on the hidden state at each iteration in the sequence length -> Generate token

Dataset is made in the form of : 
Sentence : "The cat ate the pizza on the table." 
Perspective : [Cat] 
CorrectOutput : "The pizza was delicious." 

Approaches I've tried : 
- I attempted to project the [CLS] token onto each input of the LSTM, my reasoning for this, was that adding the [CLS] token would almost be hydrating the model at each iteration, so it would be
- able to maintain the context of the original sentence even at the last generated token. However, this didn't work because the model kept converging to the same [SEP] token, likely because
- the [CLS] token wasn't meaningfully getting extracted.

- Feeding in the first token as the first input into the LSTM, I wanted to see if the model, given the correct first token, generate other tokens based on this, and be trainable.
- However, this did not work, because this input got lost after two or three passes in the same sentence. It had generated something like "I don't don't don't [SEP] [SEP]....".

Approaches to try : 
- Using the [CLS] token, but instead of using it as input, add an attention layer so whenever this is used as input, the model can learn over time what part's of the
-  [CLS] token are useful for generating tokens.

- Teacher forcing, feeding the model the correct token and using attention for all of the previously generated tokens to generate the next token. This I need to try because I believe it will yield the
- best results, however, it is hard to implement because I would have to change a lot of my code.


