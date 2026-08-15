import pytest
import training.data as data_module
from training.data import (load_translation_pairs, preprocess_sentence, preprocess_translation_pair, )


def test_preprocess_sentence() -> None:
    sentence = "  Ich   mag   Äpfel.  "

    result = preprocess_sentence(sentence)

    assert result == "Ich mag Äpfel."

def test_preprocess_translation_pair() -> None:
    german, english = preprocess_translation_pair(
        "  Guten   Morgen. ",
        " Good   morning. ",
    )

    assert german == "Guten Morgen."
    assert english == "Good morning."

@pytest.mark.parametrize(
    "sentence, exception",
    [
        ("", ValueError),
        ("   ", ValueError),
        (None, TypeError),
        (123, TypeError),
    ] )

def test_preprocess_sentence_rejects_invalid_input(
    sentence,
    exception,
) -> None:
    with pytest.raises(exception):
        preprocess_sentence(sentence)


def test_load_translation_pairs(monkeypatch) -> None:

    fake_dataset = [
        {
            "translation": {
                "de": "  Ich   mag Äpfel. ",
                "en": " I like apples. ",
            }
        },
        {
            "translation": {
                "de": "",
                "en": "Invalid pair.",
            }
        },
        {
            "translation": {
                "de": "Guten Morgen.",
                "en": "Good morning.",
            }
        },
    ]

    def fake_load_dataset(
        dataset_name,
        dataset_config,
        split,
        streaming,
    ):
        assert dataset_name == "wmt/wmt17"
        assert dataset_config == "de-en"
        assert split == "train"
        assert streaming is True

        return fake_dataset

    monkeypatch.setattr(
        data_module,
        "load_dataset",
        fake_load_dataset,
    )

    german_sentences, english_sentences = (
        load_translation_pairs(
            split="train",
            subset_size=2,
        )
    )

    assert german_sentences == [
        "Ich mag Äpfel.",
        "Guten Morgen.",
    ]

    assert english_sentences == [
        "I like apples.",
        "Good morning.",
    ]


def test_invalid_split() -> None:
    with pytest.raises(
        ValueError,
        match="split must be one of",
    ):
        load_translation_pairs(
            split="invalid",
            subset_size=10,
        )


@pytest.mark.parametrize(
    "subset_size, exception",
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
    ],
)

def test_invalid_subset_size(
    subset_size,
    exception,
) -> None:
    with pytest.raises(exception):
        load_translation_pairs(
            split="train",
            subset_size=subset_size,
        )