import torch 
from modelling.transformer import Transformer

SOURCE_VOCABULARY_SIZE = 30
TARGET_VOCABULARY_SIZE = 40
HIDDEN_SIZE = 16
NUMBER_OF_HEADS = 4
FEED_FORWARD_SIZE = 32
NUMBER_OF_ENCODER_LAYERS = 2
NUMBER_OF_DECODER_LAYERS = 2
PADDING_TOKEN_ID = 0
MAX_LENGTH = 12

def build_transformer() -> Transformer:
    return Transformer(
        source_vocabulary_size=SOURCE_VOCABULARY_SIZE,
        target_vocabulary_size=TARGET_VOCABULARY_SIZE,
        hidden_size=HIDDEN_SIZE,
        number_of_heads=NUMBER_OF_HEADS,
        feed_forward_size=FEED_FORWARD_SIZE,
        number_of_encoder_layers=NUMBER_OF_ENCODER_LAYERS,
        number_of_decoder_layers=NUMBER_OF_DECODER_LAYERS,
        source_padding_token_id=PADDING_TOKEN_ID,
        target_padding_token_id=PADDING_TOKEN_ID,
        max_length=MAX_LENGTH,
        dropout=0.0,
    )

def test_transformer_output_shape() -> None:
    model = build_transformer()
    model.eval()

    source_token_ids = torch.tensor(
        [
            [1, 4, 7, 9, 0],
            [1, 5, 8, 6, 3],
        ],
        dtype=torch.long,
    )

    target_token_ids = torch.tensor(
        [
            [1, 10, 11, 0],
            [1, 12, 13, 14],
        ],
        dtype=torch.long,
    )

    logits = model(
        source_token_ids=source_token_ids,
        target_token_ids=target_token_ids,
    )

    assert logits.shape == (
        2,
        4,
        TARGET_VOCABULARY_SIZE,
    )

    assert torch.isfinite(logits).all()

def test_transformer_supports_backward_pass() -> None:
    model = build_transformer()
    model.train()

    source_token_ids = torch.tensor(
        [
            [1, 4, 7, 0],
            [1, 5, 8, 6],
        ],
        dtype=torch.long,
    )

    complete_target_ids = torch.tensor(
        [
            [1, 10, 11, 2, 0],
            [1, 12, 13, 14, 2],
        ],
        dtype=torch.long,
    )

    decoder_input_ids = complete_target_ids[:, :-1]
    labels = complete_target_ids[:, 1:]

    logits = model(
        source_token_ids=source_token_ids,
        target_token_ids=decoder_input_ids,
    )

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=PADDING_TOKEN_ID
    )

    loss = criterion(
        logits.reshape(-1, TARGET_VOCABULARY_SIZE),
        labels.reshape(-1),
    )

    loss.backward()

    assert torch.isfinite(loss)

    assert model.output_projection.weight.grad is not None
    assert torch.isfinite(
        model.output_projection.weight.grad
    ).all()

    assert model.encoder.token_embedding.weight.grad is not None
    assert model.decoder.token_embedding.weight.grad is not None

def test_transformer_automatically_creates_masks() -> None:
    model = build_transformer()
    model.eval()

    source_token_ids = torch.tensor(
        [[1, 4, 7, 0]],
        dtype=torch.long,
    )

    target_token_ids = torch.tensor(
        [[1, 10, 11, 0]],
        dtype=torch.long,
    )

    automatic_logits = model(
        source_token_ids=source_token_ids,
        target_token_ids=target_token_ids,
    )

    source_mask = source_token_ids.ne(PADDING_TOKEN_ID)
    target_mask = target_token_ids.ne(PADDING_TOKEN_ID)

    explicit_logits = model(
        source_token_ids=source_token_ids,
        target_token_ids=target_token_ids,
        source_mask=source_mask,
        target_mask=target_mask,
    )

    assert torch.allclose(
        automatic_logits,
        explicit_logits,
        atol=1e-6,
    )