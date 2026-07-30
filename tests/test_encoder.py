import pytest
import torch

from modelling.encoder import (
    Encoder,
    EncoderLayer,
    PositionWiseFeedForward,
)

@pytest.fixture
def feed_forward() -> PositionWiseFeedForward:
    return PositionWiseFeedForward(
        hidden_size=16,
        feed_forward_size=64,
    )

@pytest.fixture
def encoder_layer() -> EncoderLayer:
    return EncoderLayer(
        hidden_size=16,
        number_of_heads=4,
        feed_forward_size=64,
        dropout=0.0,
    )


@pytest.fixture
def encoder() -> Encoder:
    return Encoder(
        vocabulary_size=100,
        hidden_size=16,
        number_of_heads=4,
        feed_forward_size=64,
        number_of_layers=2,
        padding_token_id=0,
        max_length=16,
        dropout=0.0,
    )


def test_feed_forward(
    feed_forward: PositionWiseFeedForward,
) -> None:
    x = torch.randn(2, 5, 16)
    output = feed_forward(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    with pytest.raises(ValueError, match="hidden_size=16"):
        feed_forward(torch.randn(2, 5, 8))


@pytest.mark.parametrize(
    ("hidden_size", "feed_forward_size"),
    [
        (0, 64),
        (16, 0),
    ],
)
def test_feed_forward_invalid_configuration(
    hidden_size: int,
    feed_forward_size: int,
) -> None:
    with pytest.raises(ValueError):
        PositionWiseFeedForward(
            hidden_size=hidden_size,
            feed_forward_size=feed_forward_size,
        )


def test_encoder_layer(
    encoder_layer: EncoderLayer,
) -> None:
    x = torch.randn(2, 5, 16)
    source_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )

    output = encoder_layer(x, source_mask)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()

    with pytest.raises(ValueError, match="source_mask"):
        encoder_layer(x, torch.ones(2, 4))


def test_hidden_size_must_be_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        EncoderLayer(
            hidden_size=15,
            number_of_heads=4,
            feed_forward_size=64,
        )


def test_encoder_output_and_automatic_mask(
    encoder: Encoder,
) -> None:
    encoder.eval()

    source_token_ids = torch.tensor(
        [
            [1, 2, 3, 0],
            [4, 5, 0, 0],
        ],
        dtype=torch.long,
    )
    source_mask = source_token_ids.ne(0)

    with torch.no_grad():
        automatic_output = encoder(source_token_ids)
        explicit_output = encoder(
            source_token_ids,
            source_mask,
        )

    assert automatic_output.shape == (2, 4, 16)
    assert torch.isfinite(automatic_output).all()

    torch.testing.assert_close(
        automatic_output,
        explicit_output,
    )


def test_masked_tokens_do_not_affect_valid_outputs(
    encoder: Encoder,
) -> None:
    encoder.eval()

    source_mask = torch.tensor(
        [[1, 1, 0, 0]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        first_output = encoder(
            torch.tensor([[1, 2, 0, 0]]),
            source_mask,
        )
        second_output = encoder(
            torch.tensor([[1, 2, 8, 9]]),
            source_mask,
        )

    torch.testing.assert_close(
        first_output[:, :2],
        second_output[:, :2],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("source_token_ids", "error_type", "message"),
    [
        (
            torch.ones(2, 3, 4, dtype=torch.long),
            ValueError,
            "source_token_ids",
        ),
        (
            torch.randn(2, 5),
            TypeError,
            "integer token IDs",
        ),
        (
            torch.ones(2, 17, dtype=torch.long),
            ValueError,
            "exceeds",
        ),
    ],
)
def test_encoder_rejects_invalid_inputs(
    encoder: Encoder,
    source_token_ids: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        encoder(source_token_ids)


def test_encoder_supports_backward_pass(
    encoder: Encoder,
) -> None:
    source_token_ids = torch.tensor(
        [
            [1, 2, 3, 0],
            [4, 5, 6, 7],
        ],
        dtype=torch.long,
    )

    encoder(source_token_ids).mean().backward()

    gradient = encoder.token_embedding.weight.grad

    assert gradient is not None
    torch.testing.assert_close(
        gradient[0],
        torch.zeros_like(gradient[0]),
    )

