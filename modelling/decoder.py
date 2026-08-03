import math
from typing import Optional

import torch
from torch import Tensor, nn

from .attention import MultiHeadAttention
from .encoder import PositionWiseFeedForward
from .positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer with causal self-attention, encoder-decoder cross-attention, and a feed-forward network.
    Input/output shape:  (B, target_length, hidden_size)
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
            raise ValueError( "number_of_heads must be positive, " f"got {number_of_heads}"  )                                                              
                
        if hidden_size % number_of_heads != 0:
            raise ValueError( f"hidden_size ({hidden_size}) must be divisible by " f"number_of_heads ({number_of_heads})" )
                                    
        if feed_forward_size <= 0:
            raise ValueError( "feed_forward_size must be positive, " f"got {feed_forward_size}"  )
                                   
        if not 0.0 <= dropout < 1.0:
            raise ValueError( f"dropout must be in [0, 1), got {dropout}"  )

        self.hidden_size = hidden_size

        # Decoder self-attention must prevent access to future tokens.
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            number_of_heads=number_of_heads,
            mask_future=True,
        )

        # Cross-attention can attend to the complete encoder output.
        self.cross_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            number_of_heads=number_of_heads,
            mask_future=False,
        )

        # Reuse the feed-forward network from encoder.py.
        self.feed_forward = PositionWiseFeedForward(
            hidden_size=hidden_size,
            feed_forward_size=feed_forward_size,
        )

        self.self_attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

        # Post-layer normalization.
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.cross_attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)

    def forward(
            self,
            x: Tensor,
            encoder_output: Tensor,
            target_mask: Tensor,
            source_mask: Tensor,
        ) -> Tensor: 

        """
        Apply causal self-attention, cross-attention, and feed-forward processing.
        """

        if x.ndim != 3:
            raise ValueError( "x must have shape " "(batch_size, target_length, hidden_size)" )

        if encoder_output.ndim != 3:
            raise ValueError( "encoder_output must have shape " "(batch_size, source_length, hidden_size)" )

        if x.size(-1) != self.hidden_size:
            raise ValueError( f"x must have hidden_size={self.hidden_size}, " f"but received {x.size(-1)}" )

        if encoder_output.size(-1) != self.hidden_size:
            raise ValueError( "encoder_output must have " f"hidden_size={self.hidden_size}, " f"but received {encoder_output.size(-1)}" )

        batch_size = x.size(0)
        target_length = x.size(1)
        source_length = encoder_output.size(1)

        if encoder_output.size(0) != batch_size:
            raise ValueError( "x and encoder_output must have the same batch size" )

        if encoder_output.device != x.device:
            raise ValueError( "x and encoder_output must be on the same device" )            

        target_mask = target_mask.to(
            device=x.device,
            dtype=torch.bool,
        )

        source_mask = source_mask.to(
            device=x.device,
            dtype=torch.bool,
)

        if target_mask.shape != (batch_size, target_length):
            raise ValueError( "target_mask must have shape " f"({batch_size}, {target_length}), " f"but received {tuple(target_mask.shape)}" )

        if source_mask.shape != (batch_size, source_length):
            raise ValueError( "source_mask must have shape " f"({batch_size}, {source_length}), " f"but received {tuple(source_mask.shape)}" )

        # 1. Causal decoder self-attention.
        # The target padding mask blocks padded key positions. mask_future=True internally blocks future positions.
        self_attention_output = self.self_attention(
            query=x,
            key=x,
            value=x,
            attention_mask=target_mask,
        )

        x = self.self_attention_norm( x+ self.self_attention_dropout( self_attention_output ) )

        # 2. Encoder-decoder cross-attention.       
        # Queries come from the decoder. Keys and values come from the encoder output.
        cross_attention_output = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            attention_mask=source_mask,
        )

        x = self.cross_attention_norm( x + self.cross_attention_dropout( cross_attention_output  ) )

        # 3. Position-wise feed-forward network.
        feed_forward_output = self.feed_forward(x)

        x = self.feed_forward_norm( x + self.feed_forward_dropout( feed_forward_output ) )

        return x

class Decoder(nn.Module):
    """
    Transformer decoder that converts target token IDs into contextual target hidden representations.
    Output shape:
            (B, target_length, hidden_size)
    """

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
            raise ValueError( "feed_forward_size must be positive, " f"got {feed_forward_size}" )

        if number_of_layers <= 0:
            raise ValueError(
                "number_of_layers must be positive, " f"got {number_of_layers}" )

        if max_length <= 0:
            raise ValueError( f"max_length must be positive, got {max_length}" )

        if not 0.0 <= dropout < 1.0:
            raise ValueError( f"dropout must be in [0, 1), got {dropout}" )

        if not 0 <= padding_token_id < vocabulary_size:
            raise ValueError( "padding_token_id must be within the vocabulary: "                
                f"expected a value from 0 to {vocabulary_size - 1}, "  f"got {padding_token_id}" )

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.padding_token_id = padding_token_id
        self.max_length = max_length

        self.embedding_scale = math.sqrt(hidden_size)

        self.token_embedding = nn.Embedding( num_embeddings=vocabulary_size, embedding_dim=hidden_size,
                            padding_idx=padding_token_id, )                    
            

        self.positional_encoding = PositionalEncoding( hidden_size=hidden_size, max_length=max_length, dropout=dropout, )                         
                    

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_size=hidden_size,
                    number_of_heads=number_of_heads,
                    feed_forward_size=feed_forward_size,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(
        self,
        target_token_ids: Tensor,
        encoder_output: Tensor,
        source_mask: Tensor,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode target token IDs using contextual encoder output.
        """
        if target_token_ids.ndim != 2:
            raise ValueError( "target_token_ids must have shape " "(batch_size, target_length)" )

        if encoder_output.ndim != 3:
            raise ValueError( "encoder_output must have shape " "(batch_size, source_length, hidden_size)" )

        if target_token_ids.dtype not in ( torch.int32, torch.int64, ):
            raise TypeError( "target_token_ids must contain integer token IDs, " f"but received dtype={target_token_ids.dtype}" )

        if ( (target_token_ids < 0).any() or (target_token_ids >= self.vocabulary_size).any() ):
            raise ValueError( "target_token_ids must be between " f"0 and {self.vocabulary_size - 1}"  )
                                                      
        batch_size = target_token_ids.size(0)
        target_length = target_token_ids.size(1)

        encoder_batch_size = encoder_output.size(0)
        source_length = encoder_output.size(1)

        if target_length == 0:
            raise ValueError( "target_token_ids must contain at least one token" )

        if source_length == 0:
            raise ValueError( "encoder_output must contain at least one source position"  )

        if target_length > self.max_length:
            raise ValueError( f"target length {target_length} exceeds " f"max_length={self.max_length}" )

        if encoder_batch_size != batch_size:
            raise ValueError( "target_token_ids and encoder_output must have " "the same batch size" )

        if encoder_output.size(-1) != self.hidden_size:
            raise ValueError( "encoder_output must have " f"hidden_size={self.hidden_size}, "
                                        f"but received {encoder_output.size(-1)}"  )

        if encoder_output.device != target_token_ids.device:
            raise ValueError( "target_token_ids and encoder_output must be " "on the same device"  )

        if source_mask.shape != (batch_size, source_length):
            raise ValueError( "source_mask must have shape " f"({batch_size}, {source_length}), "
                             f"but received {tuple(source_mask.shape)}"  )

        # Automatically construct the target padding mask.
        if target_mask is None:
            target_mask = target_token_ids.ne( self.padding_token_id )

        if target_mask.shape != ( batch_size, target_length, ):
            raise ValueError( "target_mask must have shape " f"{( batch_size, target_length, )}, "
                              f"but received {tuple(target_mask.shape)}"  )

        source_mask = source_mask.to( device=target_token_ids.device, dtype=torch.bool, )
        target_mask = target_mask.to( device=target_token_ids.device, dtype=torch.bool,  )

        # Detect a valid token appearing after padding in each sequence.
        invalid_right_padding = ( (~target_mask[:, :-1]) & target_mask[:, 1:]  ).any(dim=1)                                                           
        if invalid_right_padding.any().item():
            raise ValueError( "target sequences must use right padding because " "decoder self-attention is causal"  )       
                                                           
        # Completely masked sequences would produce all -inf attention rows and NaN values after softmax.
        if not source_mask.any(dim=1).all():
            raise ValueError(
                "each source sequence must contain at least one " "valid token" )

        if not target_mask.any(dim=1).all():
            raise ValueError( "each target sequence must contain at least one " "valid token" )

        # Target embedding: (B, Lt) -> (B, Lt, hidden_size)
        x = self.token_embedding(target_token_ids)

        # Scale embeddings as in the original Transformer.
        x = x * self.embedding_scale

        # Add positional information and embedding dropout.
        x = self.positional_encoding(x)

        # Pass through the stacked decoder layers.
        for layer in self.layers:
            x = layer( x=x, encoder_output=encoder_output, target_mask=target_mask, source_mask=source_mask,  )

                                               
        return x
    