# Translation Transformer

A PyTorch implementation of the Transformer architecture from
*Attention Is All You Need* for German-to-English translation.

## Status

Implemented:

- scaled dot-product and multi-head attention;
- sinusoidal positional encoding;
- Transformer encoder and decoder;
- complete encoder-decoder wrapper;
- target-vocabulary output projection;
- shared German-English tokenizer with truncation and dynamic right padding.

In progress:

- WMT17 data loading and preprocessing;
- batching and target shifting;
- training, validation, and inference;
- command-line interface.

## Project structure

```text
modelling/   Transformer components
training/    Tokenization and training pipeline
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

Uses fixed sinusoidal encodings with sine and cosine functions across embedding
dimensions.

### Encoder

Processes source token IDs using:

- scaled token embeddings;
- positional encoding;
- bidirectional self-attention;
- feed-forward networks;
- residual connections and post-layer normalization.

```text
Input:  (batch_size, source_length)
Output: (batch_size, source_length, hidden_size)
```

### Decoder

Processes target prefixes using:

- causal self-attention;
- cross-attention over encoder outputs;
- source and target padding masks;
- feed-forward networks;
- residual connections and post-layer normalization.

Target sequences must use right padding.

```text
Input:  (batch_size, target_length)
Output: (batch_size, target_length, hidden_size)
```

### Transformer wrapper

Connects the encoder and decoder and projects decoder representations to raw
target-vocabulary logits.

```text
Output:
(batch_size, target_length, target_vocabulary_size)
```

### Tokenization

Uses the `facebook/bart-base` tokenizer without loading pretrained model
weights.

It provides:

- shared German-English subword tokenization;
- automatic BOS and EOS tokens;
- PyTorch token-ID tensors;
- truncation;
- dynamic right padding;
- attention masks.

## Tests

Run all tests:

```bash
python -m pytest tests/ -v
```

Current result:

```text
34 passed
```

## Next steps

1. Test the tokenization module.
2. Load and preprocess a small WMT17 subset.
3. Implement batching and target shifting.
4. Add padding-aware loss and training loops.
5. Implement checkpointing and greedy decoding.
6. Add the command-line interface and training results.