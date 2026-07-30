# Translation Transformer

A machine translation project implementing a Transformer architecture.

## Project status

Initial project structure and environment setup.

## Structure

- `modelling/`: Transformer architecture and model components
- `training/`: Training pipeline
- `tests/`: Unit and integration tests
- `outputs/`: Generated outputs and model artifacts

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## 5. Handle `requirements.txt` and `pyproject.toml`

```text
torch
numpy
pytest


## Attention implementation

The scaled dot-product attention and multi-head attention mechanisms were
implemented from scratch using PyTorch tensor operations and linear layers.

Test coverage includes:

- self-attention;
- causal self-attention;
- cross-attention;
- padding masking;
- multi-head projection and recombination.

Test result:

```text
6 passed

### Positional Encoding implementation

Implemented reusable sinusoidal positional encoding from the *Attention Is All You Need* paper in `modelling/positional_encoding.py`.

The module:

* adds token-position information to embeddings;
* uses sine for even dimensions and cosine for odd dimensions;
* preserves the shape `(batch_size, sequence_length, hidden_size)`;
* uses `register_buffer` because positional encodings are fixed, not trainable;
* can be reused by both the encoder and decoder.
