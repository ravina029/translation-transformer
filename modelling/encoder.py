import math
from typing import Optional

import torch
from torch import Tensor, nn

from .attention import MultiHeadAttention   # Use the current package
from .positional_encoding import PositionalEncoding   # Use the current package

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network used in each Transformer encoder layer. Every sequence position is processed independently using the
    same two linear transformations.
    Shape flow: (B, L, hidden_size) -> (B, L, feed_forward_size) -> (B, L, hidden_size)

    """
    def __init__(self, hidden_size: int, feed_forward_size: int, ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError( 
                f"hidden_size must be positive, got {hidden_size}" 
                )

        if feed_forward_size <= 0:
            raise ValueError( 
                "feed_forward_size must be positive, " f"got {feed_forward_size}" 
                )

        self.hidden_size = hidden_size
        self.feed_forward_size = feed_forward_size
        

        self.linear_1 = nn.Linear( in_features=hidden_size, out_features=feed_forward_size, bias=True,  )
        self.activation = nn.ReLU()

        self.linear_2 = nn.Linear( in_features=feed_forward_size, out_features=hidden_size,bias=True,  )

    def forward(self, x:Tensor) -> Tensor:
        """
        Args: x: Tensor of shape (B, L, hidden_size).
        Returns: Tensor of shape (B, L, hidden_size).

        """

        if x.ndim != 3:
            raise ValueError( 
                "x must have shape " "(batch_size, sequence_length, hidden_size)"
            )

        if x.size(-1) != self.hidden_size:
            raise ValueError( 
                f"x must have hidden_size={self.hidden_size}, " f"but received {x.size(-1)}" 
                )

        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x


class EncoderLayer(nn.Module):
    """
       Transformer encoder layer with self-attention and a position-wise feed-forward network. Uses residual connections, dropout, 
        and post-layer normalization. Input/output shape: (batch_size, sequence_length, hidden_size)
                    
    """
    def __init__(
        self,
        hidden_size: int,
        number_of_heads: int,
        feed_forward_size: int,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if hidden_size <= 0:
            raise ValueError( f"hidden_size must be positive, got {hidden_size}" )           
                
        if number_of_heads <= 0:
            raise ValueError( "number_of_heads must be positive, " f"got {number_of_heads}" )

        if hidden_size % number_of_heads != 0:
            raise ValueError( f"hidden_size ({hidden_size}) must be divisible by "
                                            f"number_of_heads ({number_of_heads})" )
                                        
        if feed_forward_size <= 0:
            raise ValueError( "feed_forward_size must be positive, " f"got {feed_forward_size}" )  
                                                                                                         
        if not 0.0 <= dropout < 1.0:
            raise ValueError( f"dropout must be in [0, 1), got {dropout}" )
                        
        self.hidden_size = hidden_size
        self.number_of_heads = number_of_heads
        self.feed_forward_size = feed_forward_size


        # Encoder self-attention is bidirectional, so it must not use a future/causal mask.

        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            number_of_heads=number_of_heads,
            mask_future=False,
        )

        self.feed_forward = PositionWiseFeedForward(
            hidden_size=hidden_size,
            feed_forward_size=feed_forward_size,
        )
        # Dropout is applied to each sublayer's output before the residual connection.

        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

        # Post-normalization: LayerNorm(x + Sublayer(x))

        self.attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)

    def forward( self, x: Tensor, source_mask: Tensor, ) -> Tensor:
        """
        Apply encoder self-attention and feed-forward processing. x: Hidden states of shape (B, L, hidden_size).
        source_mask: Padding mask of shape (B, L). Returns: Updated hidden states with the same shape as x.
        
        """      
        if x.ndim != 3:
            raise ValueError( "x must have shape " "(batch_size, source_length, hidden_size)" )           

        if x.size(-1) != self.hidden_size:
            raise ValueError( f"x must have hidden_size={self.hidden_size}, " f"but received {x.size(-1)}" )
                            
        batch_size = x.size(0)
        source_length = x.size(1)

        if source_mask.shape != (batch_size, source_length):
            raise ValueError( "source_mask must have shape " f"({batch_size}, {source_length}), "
                 f"but received {tuple(source_mask.shape)}" 
                )                                   
                            
        # 1. Encoder self-attention using x as query, key, and value. Output shape: (B, L, hidden_size)                   
            
        attention_output = self.self_attention( query=x, key=x, value=x, attention_mask=source_mask, ) 

        # 2. First residual connection and layer normalization: LayerNorm(x + Dropout(SelfAttention(x)))
        
        x = self.attention_norm( x + self.attention_dropout(attention_output) )

        # 3. Position-wise feed-forward network
        
        feed_forward_output = self.feed_forward(x)

        # 4. Second residual connection and layer normalization : LayerNorm(x + Dropout(FeedForward(x)))
        
        x = self.feed_forward_norm( x + self.feed_forward_dropout(feed_forward_output) )

        return x

class Encoder(nn.Module): 

    """ Transformer encoder that converts source token IDs into contextual hidden representations. Output shape: (B, L, hidden_size) """
    def __init__(
        self,
        vocabulary_size: int,
        hidden_size: int,
        number_of_heads: int,
        feed_forward_size: int,
        number_of_layers: int,
        padding_token_id: int,
        max_length: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if vocabulary_size <= 0:
            raise ValueError( "vocabulary_size must be positive, " f"got {vocabulary_size}" )
                

        if hidden_size <= 0: 
            raise ValueError( f"hidden_size must be positive, got {hidden_size}" )

        if number_of_heads <= 0:
            raise ValueError( "number_of_heads must be positive, " f"got {number_of_heads}" )

        if hidden_size % number_of_heads != 0:
            raise ValueError( f"hidden_size ({hidden_size}) must be divisible by " f"number_of_heads ({number_of_heads})" )

        if feed_forward_size <= 0:
            raise ValueError( "feed_forward_size must be positive, " f"got {feed_forward_size}"  )

        if number_of_layers <= 0:
            raise ValueError( "number_of_layers must be positive, " f"got {number_of_layers}"  )

        if max_length <= 0:
            raise ValueError( f"max_length must be positive, got {max_length}" )

        if not 0.0 <= dropout < 1.0:
            raise ValueError( f"dropout must be in [0, 1), got {dropout}"  )

        if not 0 <= padding_token_id < vocabulary_size:
            raise ValueError( "padding_token_id must be within the vocabulary: " f"expected a value from 0 to {vocabulary_size - 1}, " f"got {padding_token_id}" )

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.padding_token_id = padding_token_id
        self.max_length = max_length

        # Calculate this once instead of recalculating it during every forward pass.
        self.embedding_scale = math.sqrt(hidden_size)

        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=hidden_size,
            padding_idx=padding_token_id,
        )

        self.positional_encoding = PositionalEncoding(
            hidden_size=hidden_size,
            max_length=max_length,
            dropout=dropout,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=hidden_size,
                    number_of_heads=number_of_heads,
                    feed_forward_size=feed_forward_size,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward( self, source_token_ids: Tensor, source_mask: Optional[Tensor] = None,  ) -> Tensor:
        """Encode source token IDs into contextual hidden states."""

        if source_token_ids.ndim != 2:
            raise ValueError( "source_token_ids must have shape " "(batch_size, source_length)" )

        batch_size = source_token_ids.size(0)
        source_length = source_token_ids.size(1)

        if source_length > self.max_length:
            raise ValueError( f"source length {source_length} exceeds " f"max_length={self.max_length}" )

        # The embedding layer requires integer token IDs.
        if source_token_ids.dtype not in ( torch.int32, torch.int64, ):
            raise TypeError( "source_token_ids must contain integer token IDs, " f"but received dtype={source_token_ids.dtype}" )

        # Automatically construct the source padding mask when the caller does not provide one.
        if source_mask is None:
            source_mask = source_token_ids.ne(
                self.padding_token_id
            )

        expected_mask_shape = ( batch_size, source_length, )

        if source_mask.shape != expected_mask_shape:
            raise ValueError( "source_mask must have shape " f"{expected_mask_shape}, " f"but received {tuple(source_mask.shape)}" )

        # Convert the mask to Boolean and move it to the correct device.
        source_mask = source_mask.to( device=source_token_ids.device, dtype=torch.bool, )

        # Convert source token IDs into embedding vectors: (B, L) -> (B, L, hidden_size)
        x = self.token_embedding(source_token_ids)

        # The original Transformer scales token embeddings by sqrt(d_model) before adding positional information.
        x = x * self.embedding_scale

        # Add sinusoidal positional encodings and apply dropout.
        x = self.positional_encoding(x)

        # Pass the hidden states through every encoder layer.
        for layer in self.layers:
            x = layer(
                x=x,
                source_mask=source_mask,
            )

        return x