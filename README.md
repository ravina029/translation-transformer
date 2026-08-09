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
- batch preparation and target shifting;
- end-to-end forward pass;
- padding-aware cross-entropy loss;
- backpropagation and Adam parameter updates.

In progress:

- multi-epoch training and validation;
- checkpointing;
- greedy decoding;
- command-line interface;
- final training statistics.

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

Supports self-attention, causal attention, cross-attention, padding masks, and
multiple attention heads.

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

Uses WMT17 German-English sentence pairs with minimal whitespace preprocessing.

The `facebook/bart-base` tokenizer is used without pretrained model weights for:

- shared subword tokenization;
- BOS/EOS tokens;
- truncation;
- dynamic right padding;
- attention masks.

Target sequences are shifted for autoregressive training:

```text
Target:        <BOS> I like apples <EOS>
Decoder input: <BOS> I like apples
Labels:        I like apples <EOS>
```

## Verified end-to-end pipeline

```text
WMT17
  ↓
preprocessing
  ↓
tokenization
  ↓
batch preparation
  ↓
target shifting
  ↓
Transformer
  ↓
loss
  ↓
backpropagation
  ↓
parameter update
```

Example forward-pass output:

```text
source_token_ids:  torch.Size([4, 90])
decoder_input_ids: torch.Size([4, 42])
labels:            torch.Size([4, 42])
logits:            torch.Size([4, 42, 50265])
```

## Tests

Run all tests:

```bash
python -m pytest tests/ -v
```
all 49 tests are passed 

Additional checks:

```bash
python -m training.smoke_test
python -m training.train_step
```

## Next steps

1. Implement training and validation loops.
2. Train for multiple epochs and record losses.
3. Add checkpoint saving and loading.
4. Implement greedy decoding.
5. Add the command-line interface.
6. Finalize tests and training results.