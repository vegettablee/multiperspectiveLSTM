# Multi-Perspective Text Generation with BERT + LSTM

A neural text generation system that combines BERT contextual encoding with LSTM decoding to generate text from multiple perspectives. The model uses attention mechanisms and teacher forcing for improved learning.

## 🎯 Project Overview

This project implements a novel architecture that:
- Takes an input sentence and a "perspective" prompt
- Uses BERT to encode the combined context
- Employs an LSTM decoder with attention to generate relevant text
- Learns to extract different aspects of information based on the given perspective

## 🏗️ Architecture

### Core Components

**1. BERT Encoder (`tokenizer.py`)**
- Uses pre-trained `bert-base-uncased` for contextual embeddings
- Processes input as: `"{sentence} [SEP] {perspective}"`
- Extracts CLS token as sentence-level representation

**2. Multi-Perspective LSTM Decoder (`decoder_model.py`)**
- Custom `MultiPerspectiveNN` class with attention mechanism
- Projects BERT CLS token onto LSTM hidden states
- Uses teacher forcing with attention over previous correct tokens
- Generates text autoregressively with vocabulary predictions

**3. Attention Mechanism**
- Query-Key-Value attention over previous token embeddings
- Helps model focus on relevant parts of the sequence history
- Combines attention context with LSTM hidden states for prediction

**4. Training Loop (`training_loop.py`)**
- Batch processing with gradient clipping for stability
- Comprehensive debugging and monitoring
- TensorBoard integration for loss/accuracy tracking
- Checkpoint saving for training resumption

## 🔧 Key Features

### Teacher Forcing
- Always feeds correct previous tokens (not predictions) during training
- Accelerates learning by providing ground truth context
- Model still generates predictions for loss computation

### Attention-Based Context
- Attends to all previous correct tokens in the sequence
- Provides richer context than just the previous hidden state
- Helps maintain long-term dependencies

### Gradient Stability
- Gradient clipping (max norm = 1.0) prevents exploding gradients
- Xavier initialization for stable training
- Conservative learning rate (0.0001) for stability

### Comprehensive Debugging
- Real-time monitoring of attention weights and entropy
- Gradient norm tracking and saturation detection
- Token distribution analysis to detect convergence issues
- Hidden state statistics for debugging training dynamics

## 📊 Training Metrics

The system tracks:
- **Loss**: CrossEntropyLoss across all generated tokens
- **Accuracy**: Token-level prediction accuracy
- **Token Diversity**: Distribution of predicted vocabulary
- **Attention Patterns**: Entropy and weight distribution
- **Gradient Health**: Norm monitoring and clipping effectiveness

## 🔍 Debugging Features

### Attention Analysis
- Entropy measurement (higher = more diverse attention)
- Max attention weight tracking
- Visualization of attention distribution

### Gradient Monitoring
- Pre/post clipping gradient norms
- Layer-specific gradient tracking
- Exploding gradient detection

### Vocabulary Analysis
- Token frequency distribution
- Diversity warnings for convergence issues
- Top-N most frequent predictions

## 🚀 Usage

```bash
# Start training (configured in training_loop.py)
python training_loop.py

# Monitor with TensorBoard
tensorboard --logdir runs/exp_loss_tracking
```

## 🛠️ Technical Implementation

### Data Flow
1. Input: `(sentence, perspective, correct_output)` tuples
2. BERT encoding of `"{sentence} [SEP] {perspective}"`
3. CLS token projection onto LSTM hidden states
4. Sequential token generation with attention
5. Loss computation against correct outputs

### Loss Function
- CrossEntropyLoss at each timestep
- Combines LSTM hidden state + attention context
- Accumulated across all tokens in sequence

### Model Architecture Details
- **Input**: 768-dim BERT embeddings
- **LSTM**: 2-layer with configurable hidden size
- **Attention**: Multi-head style with Q-K-V projections
- **Output**: 30,522 vocabulary predictions (BERT vocab)

## 📈 Expected Applications

This architecture could be applied to:
- **Document Summarization**: Different perspective summaries
- **Question Answering**: Perspective-guided answer generation
- **Content Creation**: Generate text from specific viewpoints
- **Information Extraction**: Extract different aspects from text

## 🔬 Research Contributions

- Novel combination of BERT encoding with LSTM decoding
- Attention mechanism over teacher-forced token history
- Comprehensive debugging framework for neural text generation
- Gradient stability techniques for training large vocabulary models

---

*This project demonstrates advanced NLP techniques including transformer integration, attention mechanisms, teacher forcing, and production-ready training practices with extensive monitoring and debugging capabilities.*