import pytest
import torch
from modelling.decoder import Decoder, DecoderLayer

@pytest.fixture
def decoder() -> Decoder:
    return Decoder(
        vocabulary_size=32,
        hidden_size=16,
        number_of_heads=4,
        feed_forward_size=32,
        number_of_layers=2,
        padding_token_id=0,
        max_length=8,
        dropout=0.0,
    )

@pytest.fixture
def sample_batch():
    target_token_ids = torch.tensor(
        [
            [1, 2, 3, 0],
            [1, 4, 5, 6],
        ],
        dtype=torch.long,
    )

    encoder_output = torch.randn(2, 5, 16)

    source_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    return target_token_ids, encoder_output, source_mask


def test_decoder_layer_output_shape():
    layer = DecoderLayer(
        hidden_size=16,
        number_of_heads=4,
        feed_forward_size=32,
        dropout=0.0,
    )

    x = torch.randn(2, 4, 16)
    encoder_output = torch.randn(2, 5, 16)
    target_mask = torch.ones(2, 4, dtype=torch.bool)
    source_mask = torch.ones(2, 5, dtype=torch.bool)

    output = layer(x, encoder_output, target_mask, source_mask)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_decoder_output_shape_and_automatic_mask(decoder, sample_batch):
    target_token_ids, encoder_output, source_mask = sample_batch

    output = decoder(
        target_token_ids=target_token_ids,
        encoder_output=encoder_output,
        source_mask=source_mask,
    )

    assert output.shape == (2, 4, 16)
    assert torch.isfinite(output).all()


def test_automatic_and_explicit_target_masks_match(decoder, sample_batch):
    target_token_ids, encoder_output, source_mask = sample_batch
    target_mask = target_token_ids.ne(0)

    automatic_output = decoder(
        target_token_ids,
        encoder_output,
        source_mask,
    )

    explicit_output = decoder(
        target_token_ids,
        encoder_output,
        source_mask,
        target_mask,
    )

    assert torch.allclose(automatic_output, explicit_output)


def test_causal_mask_blocks_future_target_tokens(decoder):
    encoder_output = torch.randn(1, 5, 16)
    source_mask = torch.ones(1, 5, dtype=torch.bool)

    target_a = torch.tensor([[1, 2, 3, 4]])
    target_b = torch.tensor([[1, 2, 3, 9]])

    output_a = decoder(target_a, encoder_output, source_mask)
    output_b = decoder(target_b, encoder_output, source_mask)

    # Changing the final token must not affect earlier positions.
    assert torch.allclose(
        output_a[:, :3],
        output_b[:, :3],
        atol=1e-6,
    )


def test_source_mask_blocks_padded_encoder_positions(decoder):
    target_token_ids = torch.tensor([[1, 2, 3]])
    source_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

    encoder_a = torch.randn(1, 5, 16)
    encoder_b = encoder_a.clone()
    encoder_b[:, 3:] = torch.randn_like(encoder_b[:, 3:]) * 1000

    output_a = decoder(target_token_ids, encoder_a, source_mask)
    output_b = decoder(target_token_ids, encoder_b, source_mask)

    assert torch.allclose(output_a, output_b, atol=1e-6)


@pytest.mark.parametrize(
    "target_token_ids",
    [
        torch.tensor([[0, 0, 1, 2]]),  # left padding
        torch.tensor([[1, 0, 2, 0]]),  # internal padding gap
    ],
)
def test_decoder_rejects_non_right_padding(decoder, target_token_ids):
    encoder_output = torch.randn(1, 4, 16)
    source_mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="right padding"):
        decoder(target_token_ids, encoder_output, source_mask)


def test_decoder_rejects_completely_masked_sequences(decoder):
    target_token_ids = torch.tensor([[1, 2, 3]])
    encoder_output = torch.randn(1, 4, 16)

    with pytest.raises(ValueError, match="source sequence"):
        decoder(
            target_token_ids,
            encoder_output,
            source_mask=torch.zeros(1, 4, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="target sequence"):
        decoder(
            torch.zeros(1, 3, dtype=torch.long),
            encoder_output,
            source_mask=torch.ones(1, 4, dtype=torch.bool),
        )


def test_backward_pass_and_padding_embedding_gradient(decoder, sample_batch):
    target_token_ids, encoder_output, source_mask = sample_batch

    output = decoder(
        target_token_ids,
        encoder_output,
        source_mask,
    )

    loss = output.square().mean()
    loss.backward()

    assert decoder.token_embedding.weight.grad is not None
    assert torch.isfinite(decoder.token_embedding.weight.grad).all()
    assert torch.equal(
        decoder.token_embedding.weight.grad[0],
        torch.zeros_like(decoder.token_embedding.weight.grad[0]),
    )
