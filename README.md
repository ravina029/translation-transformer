# Translation Transformer

A PyTorch implementation of the Transformer architecture from
*Attention Is All You Need* for German-to-English translation.

## Status

Implemented and tested:

- scaled dot-product and multi-head attention;
- sinusoidal positional encoding;
- Transformer encoder, decoder, and full encoder-decoder wrapper;
- German-English tokenization;
- WMT17 loading and preprocessing;
- mini-batch preparation and target shifting;
- padding-aware cross-entropy loss;
- backpropagation and Adam optimization;
- multi-epoch training and validation;
- best-validation checkpoint saving.

In progress:

- checkpoint loading;
- greedy decoding;
- command-line interface;
- final integration tests and evaluation.

## Project structure

```text
modelling/   Transformer components
training/    Data, tokenization, batching, training, and checkpointing
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

Projects decoder outputs to target-vocabulary logits:

```text
(batch_size, target_length, target_vocabulary_size)
```

## Data pipeline

WMT17 German-English sentence pairs are minimally preprocessed and tokenized
using `facebook/bart-base` without loading pretrained model weights.

The pipeline supports dynamic right padding, attention masks, aligned
mini-batches, and autoregressive target shifting:

```text
Target:        <BOS> I like apples <EOS>
Decoder input: <BOS> I like apples
Labels:        I like apples <EOS>
```

## Training

Training supports CUDA, Apple MPS, and CPU, with separate training and
validation passes and checkpointing based on validation loss.

Latest experiment:

```text
Training examples:   5000
Validation examples: 500
Batch size:           4
Epochs:               5
Device:               Apple MPS

Epoch 1
Training loss:   7.9270
Validation loss: 8.2466

Epoch 2
Training loss:   6.4890
Validation loss: 8.4240

Epoch 3
Training loss:   6.4539
Validation loss: 8.5801

Epoch 4
Training loss:   6.4209
Validation loss: 8.7302

Epoch 5
Training loss:   6.3753
Validation loss: 8.7075

Best validation loss: 8.2466
Best epoch:           1
Total training time:  14.77 minutes
```

Best checkpoint:

```text
outputs/checkpoints/transformer_train5000_val500.pt
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

1. Load and verify the saved checkpoint.
2. Implement greedy decoding.
3. Add German-to-English inference.
4. Add the command-line interface.
5. Finalize integration tests and documentation.