import pytest
import torch
from training.inference import greedy_decode


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ):
        return {
            "input_ids": torch.tensor([[0, 3, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
    ):
        words = {
            4: "Hello",
            5: "world",
        }

        decoded_words = [
            words[token_id.item()]
            for token_id in token_ids
            if token_id.item() in words
        ]

        return " ".join(decoded_words)


class FakeModel:
    def __init__(self, next_tokens):
        self.next_tokens = next_tokens
        self.call_count = 0

    def eval(self):
        return self

    def __call__(
        self,
        source_token_ids,
        target_token_ids,
        source_mask,
        target_mask,
    ):
        vocabulary_size = 10
        target_length = target_token_ids.size(1)

        logits = torch.zeros( 1, target_length, vocabulary_size,)

        next_token = self.next_tokens[
            min( self.call_count, len(self.next_tokens) - 1, )
        ]

        logits[:, -1, next_token] = 10.0

        self.call_count += 1

        return logits


def test_greedy_decode_stops_at_eos():
    tokenizer = FakeTokenizer()

    model = FakeModel( next_tokens=[4, 5, 2], )

    translation = greedy_decode(
        model=model,
        tokenizer=tokenizer,
        german_sentence="Hallo Welt",
        device=torch.device("cpu"),
        max_length=10,
    )

    assert translation == "Hello world"
    assert model.call_count == 3


def test_greedy_decode_stops_at_max_length():
    tokenizer = FakeTokenizer()

    model = FakeModel( next_tokens=[4], )

    translation = greedy_decode(
        model=model,
        tokenizer=tokenizer,
        german_sentence="Hallo",
        device=torch.device("cpu"),
        max_length=4,
    )

    assert translation == "Hello Hello Hello"
    assert model.call_count == 3


def test_greedy_decode_rejects_empty_sentence():
    tokenizer = FakeTokenizer()
    model = FakeModel(next_tokens=[2])

    with pytest.raises(
        ValueError, match="german_sentence must not be empty", ):
        greedy_decode(
            model=model,
            tokenizer=tokenizer,
            german_sentence="   ",
            device=torch.device("cpu"),
        )


def test_greedy_decode_rejects_invalid_max_length():
    tokenizer = FakeTokenizer()
    model = FakeModel(next_tokens=[2])

    with pytest.raises(
        ValueError, match="max_length must be greater than 1", ):
        greedy_decode(
            model=model,
            tokenizer=tokenizer,
            german_sentence="Hallo",
            device=torch.device("cpu"),
            max_length=1,
        )
