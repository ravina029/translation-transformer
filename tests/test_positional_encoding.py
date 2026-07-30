import pytest
import torch

from modelling.positional_encoding import PositionalEncoding


@pytest.fixture
def positional_encoding() -> PositionalEncoding:
    return PositionalEncoding(
        hidden_size=16,
        max_length=32,
        dropout=0.0,
    )


def test_output_shape_and_positions(
    positional_encoding: PositionalEncoding,
) -> None:
    x = torch.zeros(2, 10, 16)
    output = positional_encoding(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert not torch.equal(output[:, 0], output[:, 1])


@pytest.mark.parametrize(
    ("x", "error_message"),
    [
        (torch.zeros(2, 16), "x must have shape"),
        (torch.zeros(2, 5, 8), "hidden_size=16"),
        (torch.zeros(2, 33, 16), "exceeds"),
    ],
)
def test_invalid_inputs(
    positional_encoding: PositionalEncoding,
    x: torch.Tensor,
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        positional_encoding(x)


def test_encoding_is_registered_as_buffer(
    positional_encoding: PositionalEncoding,
) -> None:
    assert "positional_encoding" in dict(
        positional_encoding.named_buffers()
    )
    assert "positional_encoding" not in dict(
        positional_encoding.named_parameters()
    )