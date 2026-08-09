import pytest
import torch
import training.batching as batching_module
from training.batching import prepare_translation_batch


def test_prepare_translation_batch(monkeypatch) -> None:
    """
    Check that tokenized source and target sequences are converted into model-ready tensors.
    """
    source_encoding = {
        "input_ids": torch.tensor(
            [
                [0, 11, 2, 1],
                [0, 12, 13, 2],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ]
        ),
    }

    target_encoding = {
        "input_ids": torch.tensor(
            [
                [0, 21, 22, 2, 1],
                [0, 31, 32, 33, 2],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        ),
    }

    def fake_tokenize_translation_batch(
        tokenizer,
        german_sentences,
        english_sentences,
        max_length,
    ):
        return source_encoding, target_encoding

    monkeypatch.setattr(
        batching_module,
        "tokenize_translation_batch",
        fake_tokenize_translation_batch,
    )

    batch = prepare_translation_batch(
        tokenizer=object(),
        german_sentences=[
            "Hallo.",
            "Guten Morgen.",
        ],
        english_sentences=[
            "Hello.",
            "Good morning.",
        ],
        max_length=128,
    )

    # Source token IDs should remain unchanged.
    assert torch.equal( batch["source_token_ids"], source_encoding["input_ids"], )

    # Source attention mask should be converted to bool.
    assert torch.equal( batch["source_mask"], source_encoding["attention_mask"].bool(), )

    assert torch.equal( batch["decoder_input_ids"], target_encoding["input_ids"][:, :-1], )

    assert torch.equal( batch["labels"], target_encoding["input_ids"][:, 1:], )

    assert torch.equal( batch["decoder_mask"], target_encoding["attention_mask"][:, :-1].bool(), )

    assert ( batch["decoder_input_ids"].shape == batch["decoder_mask"].shape == batch["labels"].shape )


def test_rejects_mismatched_batch_sizes() -> None:
    """
    German and English batches must contain the same number of sentences.
    """
    with pytest.raises(
        ValueError, match="same number of sentences", ):
        prepare_translation_batch(
            tokenizer=object(),
            german_sentences=[
                "Hallo.",
                "Danke.",
            ],
            english_sentences=[
                "Hello.",
            ],
        )

def test_rejects_empty_batch() -> None:
    """
    Empty translation batches should not be accepted.
    """
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        prepare_translation_batch(
            tokenizer=object(),
            german_sentences=[],
            english_sentences=[],
        )




