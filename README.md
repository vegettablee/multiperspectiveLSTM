## Overview
This project takes a sentence like "The cat ate the pizza on the table" and generates a new sentence from a specified perspective (e.g., "Cat") to produce output like "The pizza was delicious."

## Architecture
Sentence → BERT Tokens/Embeddings → [CLS] Token Extraction → LSTM Hidden State → Token Generation
The system uses a BERT → LSTM sequence-to-sequence architecture for perspective-based text generation. I extract the [CLS] token as input to the LSTM's hidden state, then generate tokens based on the hidden state at each iteration in the sequence.

## Dataset Format
Sentence: "The cat ate the pizza on the table."
Perspective: [Cat] 
CorrectOutput: "The pizza was delicious."

## Technical Implementation
BERT Integration: Downloaded the model locally to handle import issues - everything needed is in tokenizer.py
LSTM Generation: Experimenting with autoregressive generation strategies (teacher forcing) and different attention mechanisms for decoding
Modern NLP Workflow: Combining transformer embeddings with recurrent models for structured text generation

## Current Research Approaches
I'm exploring attention mechanisms where the [CLS] token can learn over time what parts are useful for generating tokens, rather than directly projecting it onto each LSTM input. This should prevent the model from converging to repetitive [SEP] tokens.
Also investigating teacher forcing with attention over previously generated tokens, which should yield better results but requires significant code restructuring.

## Technologies
Python, PyTorch, BERT, LSTM, Transformers

## Setup
All dependencies and the BERT model are included. Run the main script after ensuring tokenizer.py is in the same directory.
