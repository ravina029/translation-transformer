# Translation Transformer

A PyTorch implementation of the Transformer architecture from
*Attention Is All You Need* for German-to-English translation.

## Status

Implemented and tested:

- scaled dot-product and multi-head attention;
- sinusoidal positional encoding;
- Transformer encoder, decoder, and full encoder-decoder wrapper;
- target-vocabulary output projection;
- German-English tokenization;
- WMT17 loading and preprocessing;
- mini-batch preparation and target shifting;
- end-to-end forward and backward passes;
- padding-aware cross-entropy loss;
- Adam optimization;
- multi-epoch training and validation.

In progress:

- checkpoint saving and loading;
- greedy decoding;
- command-line interface;
- final training experiments and statistics.

## Project structure

```text
modelling/   Transformer components
training/    Data, tokenization, batching, and training pipeline
tests/       Unit and integration tests
outputs/     Checkpoints, logs, and translations
```

## Setup

```bash
python3 -m venv .tvenv
source .tvenv/bin/activate
pip install -r requirements.txt
```

## Architecture

### Attention

Supports self-attention, causal attention, cross-attention, padding masks,
and multiple attention heads.

### Positional encoding

Uses fixed sinusoidal positional encodings.

### Encoder

```text
Input:  (batch_size, source_length)
Output: (batch_size, source_length, hidden_size)
```

Uses token embeddings, positional encoding, bidirectional self-attention,
feed-forward layers, residual connections, and post-layer normalization.

### Decoder

```text
Input:  (batch_size, target_length)
Output: (batch_size, target_length, hidden_size)
```

Uses causal self-attention, encoder-decoder cross-attention, feed-forward
layers, residual connections, and post-layer normalization.

### Transformer

Projects decoder outputs to:

```text
(batch_size, target_length, target_vocabulary_size)
```

## Data pipeline

WMT17 German-English sentence pairs are minimally preprocessed, tokenized using
`facebook/bart-base` without pretrained model weights, dynamically right-padded,
and converted to attention masks.

Targets are shifted for autoregressive training:

```text
Target:        <BOS> I like apples <EOS>
Decoder input: <BOS> I like apples
Labels:        I like apples <EOS>
```

## Training

The training pipeline supports:

- aligned mini-batch creation with optional shuffling;
- automatic CUDA, Apple MPS, or CPU selection;
- padding-aware cross-entropy loss;
- backpropagation and Adam optimization;
- separate training and validation passes.

Initial development run:

```text
Training examples:   100
Validation examples: 20
Batch size:           4
Epochs:               2

Epoch 1
Training loss:   10.9506
Validation loss: 10.8271

Epoch 2
Training loss:   10.6386
Validation loss: 10.6141
```

## Tests

Run all tests:

```bash
python -m pytest tests/ -v
```

Current result:

```text
49 passed
```

Additional end-to-end checks:

```bash
python -m training.smoke_test
python -m training.train_step
python -m training.train
```

## Next steps

1. Add checkpoint saving and loading.
2. Implement greedy decoding.
3. Add the command-line interface.
4. Run larger training experiments and record final statistics.
5. Finalize integration tests and documentation.