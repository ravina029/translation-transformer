# Translation Transformer

A PyTorch implementation of the Transformer architecture described in
*Attention Is All You Need*, developed for German-to-English machine translation.

## Project status

Implemented and unit-tested:

- scaled dot-product attention;
- multi-head attention;
- sinusoidal positional encoding;
- Transformer encoder;
- Transformer decoder.

In progress:

- complete encoder-decoder model;
- WMT17 German-English data pipeline;
- training and evaluation;
- translation inference;
- command-line interface.

## Project structure

```text
modelling/   Transformer model components
training/    Data processing and training code
tests/       Unit and integration tests
outputs/     Checkpoints, logs, and generated translations
```

## Setup

Create and activate the virtual environment:

```bash
python3 -m venv .tvenv
source .tvenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Implemented components

### Attention

Implemented scaled dot-product attention and multi-head attention with support
for self-attention, cross-attention, causal masking, and padding masks.

Run:

```bash
python -m pytest tests/test_attention.py -v
```

Current result:

```text
6 passed
```

### Positional encoding

Implemented sinusoidal positional encoding with dropout, shape validation, and
a non-trainable registered buffer.

Run:

```bash
python -m pytest tests/test_positional_encoding.py -v
```

Current result:

```text
5 passed
```

### Encoder

Implemented the Transformer encoder with:

- scaled token embeddings;
- sinusoidal positional encoding;
- bidirectional multi-head self-attention;
- source padding masks;
- position-wise feed-forward networks;
- residual connections;
- dropout;
- post-layer normalization;
- stacked encoder layers.

Input:

```text
source_token_ids: (batch_size, source_length)
```

Output:

```text
encoder_output: (batch_size, source_length, hidden_size)
```

Run:

```bash
python -m pytest tests/test_encoder.py -v
```

Current result:

```text
11 passed
```

### Decoder

Implemented the Transformer decoder with:

- scaled target-token embeddings;
- sinusoidal positional encoding;
- causal multi-head self-attention;
- encoder-decoder cross-attention;
- source and target padding masks;
- position-wise feed-forward networks;
- residual connections;
- dropout;
- post-layer normalization;
- stacked decoder layers.

The decoder requires right-padded target sequences.

Inputs:

```text
target_token_ids: (batch_size, target_length)
encoder_output:   (batch_size, source_length, hidden_size)
source_mask:      (batch_size, source_length)
target_mask:      (batch_size, target_length)
```

Output:

```text
decoder_output: (batch_size, target_length, hidden_size)
```

Run:

```bash
python -m pytest tests/test_decoder.py -v
```

Current result:

```text
9 passed
```

## Run all tests

```bash
python -m pytest tests/ -v
```

Current component-level test total:

```text
31 passed
```

## Next steps

1. Implement the complete encoder-decoder Transformer wrapper.
2. Add the target-vocabulary output projection.
3. Add YAML-based configuration.
4. Load and tokenize WMT17 German-English data.
5. Implement batching, target shifting, and padding-aware loss.
6. Implement training, validation, checkpointing, and inference.
7. Add a runnable command-line interface.
8. Report training statistics and example translations.