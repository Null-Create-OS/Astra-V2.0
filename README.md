# Astra V2.0

<p align="center">
  <b>A locally trained, modular AI assistant built from scratch.</b>
</p>

<p align="center">
  🚀 TensorFlow • 🧠 Transformer Architecture • 💾 Local Training • 🔧 Custom Pipeline
</p>

---

## Overview

**Astra V2.0** is a custom-built AI language model designed to explore how modern AI assistants work from the ground up.

Unlike API-based assistants, Astra is trained locally using a custom dataset, tokenizer, transformer model, and inference pipeline.

The goal of Astra is to create a lightweight, customizable AI system that can:

- Understand conversations
- Generate responses
- Learn from custom datasets
- Run locally
- Expand with new modules and abilities

---

# Features

## 🧠 Custom Transformer Model

Astra uses a decoder-style Transformer architecture inspired by modern Large Language Models.

Features:

- Multi-head attention
- Causal masking
- Token embeddings
- Feed-forward layers
- Layer normalization
- Autoregressive text generation

---

## ✍️ Custom Tokenizer

Astra includes its own tokenizer system.

Current capabilities:

- Vocabulary building
- Special tokens
- Text encoding
- Text decoding
- Dataset preprocessing

Special tokens:

```
<pad>
<unk>
<bos>
<eos>
<user>
<assistant>
```

---

## 📚 Custom Training Pipeline

Astra can train on custom datasets.

Training pipeline:

```
Raw Text
   |
   v
Tokenizer
   |
   v
Dataset Builder
   |
   v
Transformer Model
   |
   v
Checkpoint
   |
   v
Chat Interface
```

---

# Project Structure

```
Astra_V2.0/
│
├── main.py              # Main launcher
├── model.py             # Transformer architecture
├── tokenizer.py         # Astra tokenizer
├── train.py             # Training system
├── dataset.py           # Dataset processing
├── chat.py              # Chat inference
├── diagnose.py          # Model testing tools
│
├── configs/
│   └── config files
│
├── data/
│   └── training datasets
│
├── checkpoints/
│   └── saved models
│
└── README.md
```

---

# Installation

## Requirements

Python 3.10+

Recommended:

- NVIDIA GPU (optional)
- 8GB+ RAM
- TensorFlow 2.x

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```
tensorflow
numpy
tqdm
```

---

# Training Astra

Place training files inside:

```
data/
```

Example:

```
data/
 ├── conversations.txt
 ├── knowledge.txt
 └── examples.txt
```

Start training:

```bash
python main.py --mode train
```

---

# Chat With Astra

After training:

```bash
python main.py --mode chat
```

Example:

```
You: Hello Astra

Astra:
Hi! How can I help you today?
```

---

# Configuration

Astra supports different model sizes.

Example:

```python
CONFIGS = {

"tiny": {
    "seq":128,
    "d":128,
    "layers":3,
    "heads":4,
    "ff":256,
    "batch":16
}

}
```

Parameters:

| Parameter | Description |
|---|---|
| seq | Context length |
| d | Model dimension |
| layers | Transformer depth |
| heads | Attention heads |
| ff | Feed-forward size |
| batch | Training batch size |

---

# Training Example

Example output:

```
================================
        ASTRA V2.0 TRAINING
================================

Vocabulary: 2000
Training tokens: 500000
Sequences: 12000

Parameters: 900K

Epoch 1/50
Loss: 3.42

Epoch 50/50
Loss: 0.08

Checkpoint saved.
```

---

# Model Goals

Astra V2.0 is focused on:

✅ Learning language patterns  
✅ Better conversation ability  
✅ More natural responses  
✅ Efficient local operation  
✅ Expandable AI architecture  

Future goals:

- Long-term memory
- Voice interaction
- Vision capabilities
- Tool usage
- Personal assistant features
- Larger datasets
- Better reasoning

---

# Roadmap

## Astra V2.0
- [x] Transformer model
- [x] Custom tokenizer
- [x] Training pipeline
- [x] Chat interface

## Astra V2.5
- [ ] Improved dataset system
- [ ] Better response quality
- [ ] Memory module
- [ ] Faster inference

## Astra V3.0
- [ ] Multimodal support
- [ ] Voice assistant
- [ ] Agent abilities
- [ ] Advanced reasoning

---

# Why Astra?

Most AI projects start by calling an API.

Astra started differently.

The goal is to understand and build the technology behind AI assistants — from tokenization, to training, to generation.

Astra is an experiment in creating an AI system from the ground up.

---

# License

This project is currently for personal research and development.

---

<p align="center">
Built with curiosity, code, and a lot of training runs.
</p>
