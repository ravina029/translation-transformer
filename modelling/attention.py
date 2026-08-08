import math
import torch
from torch import Tensor, nn
from typing import Optional


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
        # Calculate QK^T
        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        # Scale using sqrt(d_k)
        scores = scores / math.sqrt(key.size(-1))

        # mask-shape validation
        batch_size = query.size(0)
        key_length = key.size(1)

        if attention_mask.shape != (batch_size, key_length):
            raise ValueError(
                "attention_mask must have shape  "
                f"({batch_size}, {key_length}), "
                f"but received {tuple(attention_mask.shape)}"
            )

        # Block the padded key and value positions
        blocked_mask = attention_mask.to(device=scores.device).eq(0).unsqueeze(1)
        scores = scores.masked_fill(
            blocked_mask,
            float("-inf"),
        )

        # Prevent access to the future positions when required (causal mask)
        if self.mask_future:
            query_length = query.size(-2)
            key_length = key.size(-2)

            future_mask = torch.triu(
                torch.ones(
                    query_length, key_length, dtype=torch.bool, device=scores.device
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(
                future_mask,
                float("-inf"),
            )

        # Normalize the scores over keys
        attention_weights = self.softmax(scores)

        # compute the weighted sum of the values
        output = torch.matmul(
            attention_weights,
            value,
        )

        return output


# Multihead attention implementation
class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention. Mask convention required by the tests:
        1 = valid key/value position
        0 = blocked key/value position
    """
    def __init__(
        self,
        hidden_size: int,
        number_of_heads: int,
        mask_future: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(f"hidden size must be positive, got {hidden_size}")

        if number_of_heads <= 0:
            raise ValueError(
                f"number of heads must be positive, got {number_of_heads}"
            )

        if hidden_size % number_of_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) must be divisible by,"
                f"number of heads ({number_of_heads})"
            )

        self.hidden_size = hidden_size
        self.number_of_heads = number_of_heads
        self.head_size = hidden_size // number_of_heads
        self.mask_future = mask_future

        # bias=False is required because the state dictionary contains only weight tensors.

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

        self.softmax = nn.Softmax(dim=-1)

        # attention scores are normalized over key positions

    def _split_heads(self, tensor: Tensor) -> Tensor:
        """
        convert (B,L,hidden_size) to (B,number_of_heads,L,head_size)

        """
        batch_size, sequence_length, hidden_size = tensor.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected tensor hidden size {self.hidden_size}, but got {hidden_size}"
            )

        tensor = tensor.reshape(
            batch_size, sequence_length, self.number_of_heads, self.head_size
        )

        #  apply (1,2) transpose on (B, L, H, D_h) -> (B, H, L, D_h)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        """
        covert (B, number_of_heads, L, head_size) to (B, L, hidden_size)
        """
        batch_size, number_of_heads, sequence_length, head_size = tensor.shape

        if number_of_heads != self.number_of_heads:
            raise ValueError(
                f"Expected number of heads are {self.number_of_heads}, but got {number_of_heads}"
            )

        if head_size != self.head_size:
            raise ValueError(
                f"Expected head size is {self.head_size}, but got {head_size}"
            )

        #  apply (1,2) transpose on (B, H, L, Dh) -> (B, L, H, Dh)
        tensor = tensor.transpose(1, 2).contiguous()
        # reshape to (B, L, hidden_size)
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
        Apply multihead attention to the intput query, key and values tensors.
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
                "query must have shape " "(batch_size, query_length, hidden_size)"
            )

        if key.ndim != 3:
            raise ValueError(
                "key must have shape " "(batch_size, key_length, hidden_size)"
            )

        if value.ndim != 3:
            raise ValueError(
                "value must have shape " "(batch_size, key_length, hidden_size)"
            )

        batch_size = query.size(0)
        query_length = query.size(1)
        key_length = key.size(1)

        if key.size(0) != batch_size or value.size(0) != batch_size:
            raise ValueError("query, key and value must have the same batch size")

        if value.size(1) != key_length:
            raise ValueError("key and value must have same sequence length")

        if (
            query.size(-1) != self.hidden_size
            or key.size(-1) != self.hidden_size
            or value.size(-1) != self.hidden_size
        ):
            raise ValueError(
                "The final dimension of query, key, and value must equal "
                f"hidden_size={self.hidden_size}"
            )

        # 1. Apply learned Q, K, and V projections.
        projected_query = self.query_transform(query)
        projected_key = self.key_transform(key)
        projected_value = self.value_transform(value)

        # 2. split the hidden_dimension into multilple attention heads
        # Shapes:
        #   Q: (B, H, Lq, D_h)
        #   K: (B, H, Lk, D_h)
        #   V: (B, H, Lk, D_h)

        projected_query = self._split_heads(projected_query)
        projected_key = self._split_heads(projected_key)
        projected_value = self._split_heads(projected_value)

        # 3. calculate query-key compatibility scores.
        # (B, H, Lq, D_h) @ (B, H, D_h, Lk) -> (B, H, Lq, Lk)

        scores = torch.matmul(
            projected_query,
            projected_key.transpose(-2, -1),
        )

        # 4. scale the scores by sqrt(d_k), where d_k is one head's dimention
        scores = scores / math.sqrt(self.head_size)

        # 5. apply the key, value padding mask to the scores
        # input maks: (B,L_K)
        # Expended mask : (B,1,1,L_k)
        # It broadcasts across heads and query positions so that the same mask is applied to all heads and query positions.

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, key_length):
                raise ValueError(
                    f"attention mask must have shape ({batch_size}, {key_length}),"
                    f"but received {tuple(attention_mask.shape)}"
                )
            blocked_positions = attention_mask.to(device=scores.device).eq(0)
            blocked_positions = blocked_positions[:, None, None, :]

            scores = scores.masked_fill(
                blocked_positions,
                float("-inf"),
            )

        # 6. apply the future/causal mask on the decoder self attetion.
        # True values mark the future positions that should be blocked.

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

            # shape (L_q,L_k) broadcasts to (B,H,L_q,L_k) across batch and head dimensions.
            scores = scores.masked_fill(
                future_mask,
                float("-inf"),
            )

        # 7. convert scores into attention weights(probabilities) using softmax

        attention_weights = self.softmax(scores)

        # 8. Compute weighted combination of value vectors using attention weights
        # (B, H, Lq, Lk) @ (B, H, Lk, Dh) -> (B, H, Lq, Dh)

        head_outputs = torch.matmul(
            attention_weights,
            projected_value,
        )

        # 9. Concatenate all heads
        # (B, H, Lq, Dh) -> (B, Lq, hidden_size)
        merged_output = self._merge_heads(head_outputs)

        # 10. apply the final output projection
        output = self.output_transform(merged_output)

        return output
