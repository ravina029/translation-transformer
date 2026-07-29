import math
import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """
    Add sinussoidal positional information to the token embeddings.
    This is a standard technique used in transformer models to provide information about the position of tokens in a sequence.
    """

    def __init__(
        self, hidden_size: int, max_length: int = 512, dropout: float = 0.1
    ) -> None:

        super().__init__()

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")

        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout)

        # Position numbers: [[0], [1], [2], ...., [max_legth-1]
        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Frequencies used by sine and cosine.
        div_term = torch.exp(
            torch.arange(
                0,
                hidden_size,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / hidden_size)
        )

        # prepares storage for the positional encoding matrix.
        positional_encoding = torch.zeros(
            max_length,
            hidden_size,
        )

        # Even feature dimensions use sine.
        positional_encoding[:, 0::2] = torch.sin(position * div_term)

        # Odd feature dimensions use cosine.
        positional_encoding[:, 1::2] = torch.cos(
            position * div_term[: positional_encoding[:, 1::2].shape[1]]
        )

        # Add a batch dimension: (max_length, hidden_size) -> (1, max_length, hidden_size)
        positional_encoding = positional_encoding.unsqueeze(0)

        # A buffer moves with the model to CPU, MPS, or GPU, it is not a trainable parameter.
        self.register_buffer(
            "positional_encoding",
            positional_encoding,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape " "(batch_size, sequence_length, hidden_size)"
            )
        if x.size(-1) != self.hidden_size:
            raise ValueError(
                f"x must have hidden_size={self.hidden_size}, "
                f"but received {x.size(-1)}"
            )

        sequence_length = x.size(1)

        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_length={self.max_length}"
            )

        positions = self.positional_encoding[:, :sequence_length].to(dtype=x.dtype)

        return self.dropout(x + positions)
