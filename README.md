# Translation Transformer

A PyTorch implementation of a small German-to-English Transformer following *Attention Is All You Need*.

The project focuses on implementing, debugging, and training the Transformer architecture from scratch rather than achieving state-of-the-art translation quality.

## Features

- Scaled dot-product attention implemented from scratch
- Multi-head attention implemented from scratch
- Padding and causal attention masks
- Sinusoidal positional encoding
- Transformer encoder and decoder
- Encoder-decoder cross-attention
- WMT17 German-English data loading
- `facebook/bart-base` tokenizer without pretrained model weights
- Dynamic right padding and target shifting
- Cross-entropy loss with padding ignored
- Adam optimization
- Training and validation loops
- Best-validation checkpoint saving and loading
- Greedy autoregressive decoding
- CUDA, Apple MPS, and CPU support

## Project Structure

```text
translation-transformer/
├── modelling/
│   ├── attention.py
│   ├── positional_encoding.py
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
├── training/
│   ├── data.py
│   ├── tokenization.py
│   ├── batching.py
│   ├── loss.py
│   ├── train_step.py
│   ├── train.py
│   ├── checkpoint.py
│   └── inference.py
├── tests/
├── outputs/
│   └── checkpoints/
├── translate.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .tvenv
source .tvenv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Model

The Transformer uses the standard encoder-decoder architecture.

Scaled dot-product attention is computed as:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

The encoder contains bidirectional self-attention and position-wise feed-forward layers.

The decoder contains:

1. causal self-attention;
2. encoder-decoder cross-attention;
3. position-wise feed-forward layers.

Residual connections and post-layer normalization are used throughout.

The final decoder representation is projected to target-vocabulary logits:

```text
(batch_size, target_length, hidden_size)
                    ↓
(batch_size, target_length, target_vocabulary_size)
```

## Data and Tokenization

The project uses the WMT17 German-English translation dataset.

Tokenization uses:

```text
facebook/bart-base
```

Only the tokenizer is used. No pretrained BART model weights are loaded.

```text
Vocabulary size: 50265
BOS token ID:    0
PAD token ID:    1
EOS token ID:    2
Maximum length:  128
```

Targets are shifted for autoregressive training:

```text
Target:        <BOS> I like apples <EOS>
Decoder input: <BOS> I like apples
Labels:        I like apples <EOS>
```

## Training

Run training with:

```bash
python -m training.train
```

The training pipeline automatically selects CUDA, Apple MPS, or CPU depending on availability.

A development training run used the following configuration:

```text
Training examples:      5000
Validation examples:     500
Batch size:                 4
Epochs:                     5

Hidden size:              128
Attention heads:            4
Feed-forward size:        256
Encoder layers:             2
Decoder layers:             2
Dropout:                   0.1
Learning rate:            1e-4
Maximum length:            128

Device:              Apple MPS
```

### Training Results

```text
Epoch   Training Loss   Validation Loss
1       7.9270          8.2466
2       6.4890          8.4240
3       6.4539          8.5801
4       6.4209          8.7302
5       6.3753          8.7075
```

```text
Best validation loss: 8.2466
Best epoch:           1
Total training time:  14.77 minutes
```

The best checkpoint is saved under:

```text
outputs/checkpoints/
```

Checkpoint files are excluded from version control because of their size.

## Checkpointing

The training pipeline saves a checkpoint whenever the validation loss improves.

Saved checkpoints contain:

- model parameters
- optimizer state
- training epoch
- training loss
- validation loss

Checkpoints can later be loaded for evaluation or inference.

## Inference

Greedy autoregressive decoding is implemented in:

```text
training/inference.py
```

Generation starts with the BOS token and predicts one token at a time using the token with the highest output logit:

```text
<BOS>
<BOS> token_1
<BOS> token_1 token_2
<BOS> token_1 token_2 token_3
...
```

Generation stops when the EOS token is predicted or the maximum sequence length is reached.

## Tests

Run the complete test suite with:

```bash
python -m pytest tests/ -v
```

Additional pipeline checks can be run with:

```bash
python -m training.smoke_test
python -m training.train_step
```

The tests cover:

- scaled dot-product attention
- multi-head attention
- padding masks
- causal masks
- positional encoding
- encoder behavior
- decoder behavior
- encoder-decoder cross-attention
- batching
- target shifting
- Transformer output shapes
- loss computation
- training behavior

## Limitations

The model is intentionally small and is trained from scratch on a limited subset of WMT17 so that training remains feasible on local hardware.

Translation quality is therefore limited. The purpose of the project is to demonstrate correct Transformer implementation, training, debugging, checkpointing, and inference rather than competitive machine-translation performance.

## Remaining Work

- Finalize the training command-line interface
- Finalize the translation entry point
- Add inference tests
- Run final integration tests