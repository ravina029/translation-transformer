from torch import Tensor, nn

from modelling.decoder import Decoder
from modelling.encoder import Encoder


class Transformer(nn.Module):
    """
    Complete encoder-decoder Transformer where:

    1. The model encodes the source sequence.
    2. The model decodes the target-token prefix using causal self-attention.
    3. The model projects each decoder representation to target-vocabulary logits.
    """
    def __init__(self,
        source_vocabulary_size: int,
        target_vocabulary_size: int,
        hidden_size: int,
        number_of_heads: int,
        feed_forward_size: int,
        number_of_encoder_layers: int,
        number_of_decoder_layers: int,
        source_padding_token_id: int,
        target_padding_token_id: int,
        max_length: int = 128,
        dropout: float = 0.1,
        tie_target_embedding_and_output: bool = False,
    ) -> None:

        super().__init__()

        if source_vocabulary_size <= 0:
            raise ValueError( f"source_vocabulary_size must be positive, got {source_vocabulary_size}" )
        
        if target_vocabulary_size <= 0:
            raise ValueError( f"target_vocabulary_size must be positive, got {target_vocabulary_size}" )

        if hidden_size <= 0:
            raise ValueError( f"hidden_size must be positive, got {hidden_size}" )

        if not 0 <= source_padding_token_id < source_vocabulary_size :
            raise ValueError( f"source_padding_token_id must be inside the source vocabulary size, got {source_padding_token_id}" )

        if not 0 <= target_padding_token_id < target_vocabulary_size:
            raise ValueError( f"target_padding_token_id must be inside the target vocabulary, got {target_padding_token_id}" )

        self.source_padding_token_id = source_padding_token_id
        self.target_padding_token_id = target_padding_token_id
        self.target_vocabulary_size = target_vocabulary_size

        self.encoder = Encoder(
            vocabulary_size = source_vocabulary_size,
            hidden_size = hidden_size,
            number_of_heads = number_of_heads,
            feed_forward_size = feed_forward_size,
            number_of_layers = number_of_encoder_layers,
            padding_token_id = source_padding_token_id,
            max_length = max_length,
            dropout = dropout, 
        )

        self.decoder= Decoder(
            vocabulary_size = target_vocabulary_size,
            hidden_size = hidden_size,
            number_of_heads = number_of_heads,
            feed_forward_size = feed_forward_size,
            number_of_layers = number_of_decoder_layers,
            padding_token_id = target_padding_token_id,
            max_length = max_length,
            dropout = dropout,
        )

        # Converts every decoder hidden vector of size hidden_size into one score for every token in the target vocabulary.
        self.output_projection = nn.Linear(
            in_features = hidden_size,
            out_features = target_vocabulary_size,
            bias = False, 
        )

        # Optional parameter sharing between the target-token embedding and the output projection.
        if tie_target_embedding_and_output:
            self.output_projection.weight = self.decoder.token_embedding.weight

    def forward(
        self,
        source_token_ids: Tensor,
        target_token_ids: Tensor,
        source_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """
        This will Run the complete Transformer.
        Returns: Raw target-vocabulary logits with shape: (batch_size, target_length, target_vocabulary_size)           
        """

        if source_mask is None:
            source_mask = source_token_ids.ne( self.source_padding_token_id )

        if target_mask is None:
            target_mask = target_token_ids.ne( self.target_padding_token_id )

        encoder_output = self.encoder(
            source_token_ids=source_token_ids,
            source_mask=source_mask,
        )

        decoder_output = self.decoder(
            target_token_ids=target_token_ids,
            encoder_output=encoder_output,
            source_mask=source_mask,
            target_mask=target_mask,
        )

        logits = self.output_projection(decoder_output)

        return logits