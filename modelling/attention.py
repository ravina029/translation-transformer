import math

import torch
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(self, mask_future: bool = False) -> None:
        super().__init__()
        self.mask_future = mask_future
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        # 1. Calculate QK^T.
        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        # 2. Scale using sqrt(d_k).
        scores = scores / math.sqrt(key.size(-1))

        # 3. Block padded key/value positions.
        blocked_mask = attention_mask.eq(0).unsqueeze(1)
        scores = scores.masked_fill(
            blocked_mask,
            float("-inf"),
        )

        # 4. Prevent access to future positions when required.
        if self.mask_future:
            query_length = query.size(-2)
            key_length = key.size(-2)

            future_mask = torch.triu(
                torch.ones(
                    query_length,
                    key_length,
                    dtype=torch.bool,
                    device=scores.device,
                ),
                diagonal=1,
            )

            scores = scores.masked_fill(
                future_mask,
                float("-inf"),
            )

        # 5. Normalize over keys.
        attention_weights = self.softmax(scores)

        # 6. Form the weighted value mixture.
        output = torch.matmul(
            attention_weights,
            value,
        )

        return output


    import math
from typing import Optional

import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention.

    Expected input shapes:
        query: (batch_size, query_length, hidden_size)
        key:   (batch_size, key_length, hidden_size)
        value: (batch_size, key_length, hidden_size)

        attention_mask: (batch_size, key_length)

    Mask convention required by the tests:
        1 = valid key/value position
        0 = blocked key/value position

    Output shape:
        (batch_size, query_length, hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        number_of_heads: int,
        mask_future: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )

        if number_of_heads <= 0:
            raise ValueError(
                "number_of_heads must be positive, "
                f"got {number_of_heads}"
            )

        if hidden_size % number_of_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"number_of_heads ({number_of_heads})"
            )

        self.hidden_size = hidden_size
        self.number_of_heads = number_of_heads
        self.head_size = hidden_size // number_of_heads
        self.mask_future = mask_future

        # These names must exactly match the supplied test state dictionary.
        # bias=False is required because the state dictionary contains
        # only weight tensors.
        self.query_transform = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.key_transform = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.value_transform = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.output_transform = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        # Attention probabilities are normalized over key positions.
        self.softmax = nn.Softmax(dim=-1)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        """
        Convert:
            (B, L, hidden_size)

        into:
            (B, number_of_heads, L, head_size)
        """
        batch_size, sequence_length, hidden_size = tensor.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected tensor hidden size {self.hidden_size}, "
                f"but received {hidden_size}"
            )

        tensor = tensor.reshape(
            batch_size,
            sequence_length,
            self.number_of_heads,
            self.head_size,
        )

        # (B, L, H, Dh) -> (B, H, L, Dh)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        """
        Convert:
            (B, number_of_heads, L, head_size)

        into:
            (B, L, hidden_size)
        """
        batch_size, number_of_heads, sequence_length, head_size = (
            tensor.shape
        )

        if number_of_heads != self.number_of_heads:
            raise ValueError(
                f"Expected {self.number_of_heads} heads, "
                f"but received {number_of_heads}"
            )

        if head_size != self.head_size:
            raise ValueError(
                f"Expected head size {self.head_size}, "
                f"but received {head_size}"
            )

        # (B, H, L, Dh) -> (B, L, H, Dh)
        tensor = tensor.transpose(1, 2).contiguous()

        # (B, L, H, Dh) -> (B, L, hidden_size)
        return tensor.reshape(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply multi-head attention.

        Args:
            query:
                Shape (B, Lq, hidden_size).

            key:
                Shape (B, Lk, hidden_size).

            value:
                Shape (B, Lk, hidden_size).

            attention_mask:
                Shape (B, Lk).

                Values:
                    1 = valid key/value position
                    0 = blocked key/value position

        Returns:
            Tensor of shape (B, Lq, hidden_size).
        """
        if query.ndim != 3:
            raise ValueError(
                "query must have shape "
                "(batch_size, query_length, hidden_size)"
            )

        if key.ndim != 3:
            raise ValueError(
                "key must have shape "
                "(batch_size, key_length, hidden_size)"
            )

        if value.ndim != 3:
            raise ValueError(
                "value must have shape "
                "(batch_size, key_length, hidden_size)"
            )

        batch_size = query.size(0)
        query_length = query.size(1)
        key_length = key.size(1)

        if key.size(0) != batch_size or value.size(0) != batch_size:
            raise ValueError(
                "query, key, and value must have the same batch size"
            )

        if value.size(1) != key_length:
            raise ValueError(
                "key and value must have the same sequence length"
            )

        if query.size(-1) != self.hidden_size:
            raise ValueError(
                f"query last dimension must be {self.hidden_size}"
            )

        if key.size(-1) != self.hidden_size:
            raise ValueError(
                f"key last dimension must be {self.hidden_size}"
            )

        if value.size(-1) != self.hidden_size:
            raise ValueError(
                f"value last dimension must be {self.hidden_size}"
            )

        # -------------------------------------------------------------
        # 1. Apply learned Q, K, and V projections.
        # -------------------------------------------------------------
        projected_query = self.query_transform(query)
        projected_key = self.key_transform(key)
        projected_value = self.value_transform(value)

        # -------------------------------------------------------------
        # 2. Split the hidden dimension into attention heads.
        #
        # Shapes:
        #   Q: (B, H, Lq, Dh)
        #   K: (B, H, Lk, Dh)
        #   V: (B, H, Lk, Dh)
        # -------------------------------------------------------------
        projected_query = self._split_heads(projected_query)
        projected_key = self._split_heads(projected_key)
        projected_value = self._split_heads(projected_value)

        # -------------------------------------------------------------
        # 3. Calculate query-key compatibility scores.
        #
        # (B, H, Lq, Dh) @ (B, H, Dh, Lk)
        # -> (B, H, Lq, Lk)
        # -------------------------------------------------------------
        scores = torch.matmul(
            projected_query,
            projected_key.transpose(-2, -1),
        )

        # -------------------------------------------------------------
        # 4. Scale by sqrt(d_k), where d_k is one head's dimension.
        # -------------------------------------------------------------
        scores = scores / math.sqrt(self.head_size)

        # -------------------------------------------------------------
        # 5. Apply the key/value padding mask.
        #
        # Input mask:
        #   (B, Lk)
        #
        # Expanded mask:
        #   (B, 1, 1, Lk)
        #
        # It broadcasts across heads and query positions.
        # -------------------------------------------------------------
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, key_length):
                raise ValueError(
                    "attention_mask must have shape "
                    f"({batch_size}, {key_length}), "
                    f"but received {tuple(attention_mask.shape)}"
                )

            blocked_positions = attention_mask.to(
                device=scores.device
            ).eq(0)

            blocked_positions = blocked_positions[:, None, None, :]

            scores = scores.masked_fill(
                blocked_positions,
                float("-inf"),
            )

        # -------------------------------------------------------------
        # 6. Apply the causal/future mask for decoder self-attention.
        #
        # True values mark future positions that must be blocked.
        # -------------------------------------------------------------
        if self.mask_future:
            future_mask = torch.triu(
                torch.ones(
                    query_length,
                    key_length,
                    dtype=torch.bool,
                    device=scores.device,
                ),
                diagonal=1,
            )

            # Shape (Lq, Lk) broadcasts to (B, H, Lq, Lk).
            scores = scores.masked_fill(
                future_mask,
                float("-inf"),
            )

        # -------------------------------------------------------------
        # 7. Convert scores into attention probabilities.
        # -------------------------------------------------------------
        attention_weights = self.softmax(scores)

        # -------------------------------------------------------------
        # 8. Compute weighted combinations of value vectors.
        #
        # (B, H, Lq, Lk) @ (B, H, Lk, Dh)
        # -> (B, H, Lq, Dh)
        # -------------------------------------------------------------
        head_outputs = torch.matmul(
            attention_weights,
            projected_value,
        )

        # -------------------------------------------------------------
        # 9. Concatenate all heads.
        #
        # (B, H, Lq, Dh) -> (B, Lq, hidden_size)
        # -------------------------------------------------------------
        merged_output = self._merge_heads(head_outputs)

        # -------------------------------------------------------------
        # 10. Apply the final output projection.
        # -------------------------------------------------------------
        output = self.output_transform(merged_output)

        return output