# Translation Transformer

A PyTorch implementation of a small German-to-English Transformer following the architecture introduced in *Attention Is All You Need*.

The project focuses on implementing, debugging, training, and running a Transformer rather than achieving state-of-the-art translation quality.

## Features

* Scaled dot-product attention implemented from scratch
* Multi-head attention implemented from scratch
* Padding and causal attention masks
* Sinusoidal positional encoding
* Transformer encoder and decoder
* Encoder-decoder cross-attention
* WMT17 German-English data loading
* `facebook/bart-base` tokenizer for text-to-token conversion, without pretrained BART model weights
* Dynamic right padding and autoregressive target shifting
* Padding-aware cross-entropy loss
* Adam optimization
* Training and validation loops
* Best-validation checkpoint saving and loading
* Configurable training CLI
* Greedy autoregressive decoding
* German-to-English translation CLI
* CUDA, Apple MPS, and CPU support

## Project Structure
```text
translation-transformer/
├── modelling/          # Transformer architecture components
├── training/           # Data, tokenization, batching, training, validation, checkpointing, and inference
├── tests/              # Unit and integration tests
├── outputs/
│   └── checkpoints/    # Saved model checkpoints
├── translate.py        # German-to-English inference CLI
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup
Create and activate a virtual environment:

```bash
python3 -m venv .tvenv
source .tvenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model
The model uses a standard Transformer encoder-decoder architecture.

Scaled dot-product attention is computed as:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

The encoder uses bidirectional self-attention and position-wise feed-forward layers.

The decoder uses:

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

The implementation uses a reduced model size, a limited WMT17 subset, and a simplified Adam training configuration so that training remains feasible on local hardware. It is therefore not intended as an exact reproduction of the original paper's training setup.

## Data and Tokenization
The project uses the WMT17 German-English translation dataset (`wmt/wmt17`, configuration `de-en`), with German as the source language and English as the target language.

Tokenization uses:

```text
facebook/bart-base
```

Only the pretrained BART tokenizer is used for text-to-token conversion. No pretrained BART model weights are loaded. The Transformer architecture and all model parameters are implemented and trained from scratch.

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
Training can be run with the default configuration:

```bash
python -m training.train
```

Available command-line options:

```bash
python -m training.train --help
```

The main configurable options are:

```text
--train-size
--validation-size
--batch-size
--epochs
--learning-rate
```

Example:

```bash
python -m training.train \
    --train-size 5000 \
    --validation-size 500 \
    --batch-size 4 \
    --epochs 5 \
    --learning-rate 1e-4
```

The training pipeline automatically selects CUDA, Apple MPS, or CPU depending on availability.

Checkpoints are named automatically from the requested dataset sizes, for example:

```text
outputs/checkpoints/transformer_train5000_val500.pt
```

## Training Experiment
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

### Results
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

Validation loss was lowest after the first epoch and increased during later epochs while training loss continued to decrease, indicating that this small development configuration begins to overfit quickly.

## Checkpointing
The best checkpoint is saved whenever validation loss improves.

A checkpoint contains:

* model parameters
* optimizer state
* training epoch
* training loss
* validation loss

Checkpoint files are excluded from version control because of their size.

Saved checkpoints can be loaded later for inference or continued experimentation.

When loading a checkpoint, the Transformer must be constructed using the same architecture and tokenizer vocabulary that were used during training.

## Translation
Greedy autoregressive decoding is implemented in `training/inference.py`.

Generation starts with the BOS token and predicts one token at a time using the token with the highest output logit. Generation stops when EOS is predicted or the maximum sequence length is reached.

Run translation with:

```bash
python translate.py "Wiederaufnahme der Sitzungsperiode"
```

By default, the script looks for the following checkpoint:

```text
outputs/checkpoints/transformer_train5000_val500.pt
```

Run the training command first to create a checkpoint, or provide an existing checkpoint with `--checkpoint`.

A different checkpoint can be selected with:

```bash
python translate.py \
    "Wiederaufnahme der Sitzungsperiode" \
    --checkpoint outputs/checkpoints/model.pt
```

View translation CLI options with:

```bash
python translate.py --help
```

## Tests
Run the complete test suite with:

```bash
python -m pytest tests/ -v
```

The tests cover:

* scaled dot-product attention
* multi-head attention
* padding and causal masks
* positional encoding
* encoder and decoder behavior
* encoder-decoder cross-attention
* batching and target shifting
* Transformer output shapes
* loss computation
* backward propagation
* greedy autoregressive decoding

A final local test run using Python 3.11.4 produced:

```text
61 passed in 6.78s
```

Additional end-to-end pipeline validation can be run with:

```bash
python -m training.smoke_test
```

The smoke test loads a small real WMT17 batch, prepares the model inputs, runs a complete Transformer forward pass, and verifies the output shape and numerical finiteness. The final smoke-test run completed successfully.

## Limitations
The model is intentionally small and is trained from scratch on a limited subset of WMT17 so that training remains feasible on local hardware.

The shared BART tokenizer has a 50,265-token vocabulary, which is large relative to the 5,000 training sentence pairs used in the development run. Consequently, the model receives relatively little supervision for learning the target-token distribution, and high translation quality is not expected.

The experiment is intended to demonstrate the Transformer implementation, masking, batching, training, validation, checkpointing, and autoregressive inference rather than competitive machine-translation performance.

## Possible Improvements
Possible extensions beyond the current development configuration include:

* training on a larger subset of WMT17;
* training a smaller German-English subword tokenizer instead of using the 50,265-token BART vocabulary;
* using the Transformer learning-rate warmup and scheduling strategy from the original paper;
* experimenting with embedding/output-weight tying and larger model configurations;
* adding beam-search decoding for improved inference quality.
