import torch 
from training.tokenization import (build_tokenizer, tokenize_translation_batch)


def test_translation_tokenizer() -> None:
    tokenizer= build_tokenizer()

    german_sentences = [
        "Ich mag Äpfel.",
        "Das ist ein längerer deutscher Beispielsatz.",
    ]

    english_sentences = [
        "I like apples.",
        "This is a longer English example sentence.",
    ]

    source, target = tokenize_translation_batch(
        tokenizer=tokenizer,
        german_sentences=german_sentences,
        english_sentences=english_sentences,
        max_length=32,
    )

    assert source["input_ids"].shape == source[ "attention_mask" ].shape

    assert target["input_ids"].shape == target[ "attention_mask" ].shape

    assert source["input_ids"].dtype == torch.long
    assert target["input_ids"].dtype == torch.long

    # Check that padding is only on the right.
    for encoding in (source, target):
        mask = encoding["attention_mask"].bool()

        invalid_right_padding = ( (~mask[:, :-1]) & mask[:, 1:] ).any()

        assert not invalid_right_padding.item()


    # Every valid sequence should begin with BOS and end with EOS.
    for encoding in (source, target):
        input_ids = encoding["input_ids"]
        mask = encoding["attention_mask"].bool()

        for sequence, sequence_mask in zip( input_ids, mask, strict=True, ):
            valid_token_ids = sequence[sequence_mask]
            assert (valid_token_ids[0].item() == tokenizer.bos_token_id )
            assert ( valid_token_ids[-1].item() == tokenizer.eos_token_id )

