# Translation Transformer

A PyTorch implementation of the Transformer architecture from
*Attention Is All You Need* for German-to-English translation.

## Status

Implemented and tested:

- scaled dot-product and multi-head attention;
- sinusoidal positional encoding;
- Transformer encoder and decoder;
- complete encoder-decoder wrapper;
- target-vocabulary output projection;
- German-English tokenization with truncation and dynamic right padding;
- WMT17 German-English data loading and preprocessing.

In progress:

- dataset-to-tokenizer integration;
- batching and target shifting;
- training and validation;
- translation inference;
- command-line interface.

## Project structure

```text
modelling/   Transformer components
training/    Tokenization, data processing, and training pipeline
tests/       Unit and integration tests
outputs/     Checkpoints, logs, and translations
```

## Setup

```bash
python3 -m venv .tvenv
source .tvenv/bin/activate
pip install -r requirements.txt
```

## Model architecture

### Attention

Supports self-attention, causal attention, cross-attention, padding masks, and
multiple attention heads.

### Positional encoding

Uses fixed sinusoidal positional encodings across embedding dimensions.

### Encoder

Processes source token IDs using token embeddings, positional encoding,
bidirectional self-attention, feed-forward networks, residual connections,
and post-layer normalization.

```text
Input:  (batch_size, source_length)
Output: (batch_size, source_length, hidden_size)
```

### Decoder

Processes target prefixes using causal self-attention, cross-attention over
encoder outputs, feed-forward networks, residual connections, and
post-layer normalization.

Target sequences use right padding.

```text
Input:  (batch_size, target_length)
Output: (batch_size, target_length, hidden_size)
```

### Transformer wrapper

Connects the encoder and decoder and projects decoder representations to raw
target-vocabulary logits.

```text
Output: (batch_size, target_length, target_vocabulary_size)
```

### Tokenization

Uses the `facebook/bart-base` tokenizer without loading pretrained model
weights.

Supports:

- shared German-English subword tokenization;
- automatic BOS and EOS tokens;
- PyTorch token-ID tensors;
- sequence truncation;
- dynamic right padding;
- attention masks;
- input validation.

### Data loading and preprocessing

Loads German-English sentence pairs from the WMT17 dataset.

The data pipeline:

- supports train, validation, and test splits;
- streams examples from the dataset;
- extracts aligned German-English translation pairs;
- validates sentence pairs;
- removes empty or malformed examples;
- normalizes leading, trailing, and repeated whitespace;
- supports small configurable subsets for development and testing.

Text preprocessing is intentionally minimal. Punctuation, capitalization,
German characters, and sentence content are preserved for tokenization.

## Tests

Run all tests:

```bash
python -m pytest tests/ -v
```

Current result:

```text
UPDATE_AFTER_RUNNING_FULL_TEST_SUITE
```

## Next steps

1. Connect WMT17 data loading to the tokenization pipeline.
2. Implement batch preparation and target shifting.
3. Pass a real WMT17 batch through the Transformer.
4. Add padding-aware loss and training/validation loops.
5. Implement checkpointing and greedy decoding.
6. Add the command-line interface and training results.