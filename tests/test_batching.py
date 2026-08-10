import pytest
import torch
import training.batching as batching_module
from training.batching import create_minibatches, prepare_translation_batch

def test_create_minibatches() -> None:
    german_sentences = [
        "Deutsch 1",
        "Deutsch 2",
        "Deutsch 3",
        "Deutsch 4",
        "Deutsch 5",
    ]

    english_sentences = [
        "English 1",
        "English 2",
        "English 3",
        "English 4",
        "English 5",
    ]

    batches = create_minibatches(
        german_sentences=german_sentences,
        english_sentences=english_sentences,
        batch_size=2,
        shuffle=False,
    )

    assert len(batches) == 3
    assert batches[0] == (
        ["Deutsch 1", "Deutsch 2"],
        ["English 1", "English 2"],
    )
    assert batches[1] == (
        ["Deutsch 3", "Deutsch 4"],
        ["English 3", "English 4"],
    )
    assert batches[2] == (
        ["Deutsch 5"],
        ["English 5"],
    )

def test_create_minibatches_preserves_alignment(monkeypatch) -> None:
    german_sentences = ["Deutsch 0", "Deutsch 1", "Deutsch 2"]
    english_sentences = ["English 0", "English 1", "English 2"]

    def fake_shuffle(indices) -> None:
        indices.reverse()

    monkeypatch.setattr( batching_module.random,"shuffle",fake_shuffle, )
        
    batches = create_minibatches(
        german_sentences=german_sentences,
        english_sentences=english_sentences,
        batch_size=2,
        shuffle=True, )

    assert batches == [ 
        ( ["Deutsch 2", "Deutsch 1"],
        ["English 2", "English 1"], ),

        ( ["Deutsch 0"],
            ["English 0"],),]   

def test_create_minibatches_rejects_mismatched_sizes() -> None:
    with pytest.raises( ValueError, match="same number of sentences", ):
        create_minibatches(
            german_sentences=["Hallo.", "Danke."],
            english_sentences=["Hello."],
            batch_size=2,
        )

@pytest.mark.parametrize(
    "batch_size, exception", [ (0, ValueError), (-1, ValueError), (1.5, TypeError), ], )

def test_create_minibatches_rejects_invalid_batch_size( batch_size, exception, ) -> None:
    with pytest.raises(exception):
        create_minibatches(
            german_sentences=["Hallo."],
            english_sentences=["Hello."],
            batch_size=batch_size,
        )

def test_create_minibatches_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="must not be empty", ):
        create_minibatches(
            german_sentences=[],
            english_sentences=[],
            batch_size=2,
        )

def test_prepare_translation_batch(monkeypatch) -> None:
    source_encoding = { 
        "input_ids": torch.tensor   ([  [0, 11, 2, 1],
                                        [0, 12, 13, 2], ]),
            
        "attention_mask": torch.tensor([ [1, 1, 1, 0],
                                        [1, 1, 1, 1], ]),            
    }

    target_encoding = {
        "input_ids": torch.tensor( [[0, 21, 22, 2, 1],
                                    [0, 31, 32, 33, 2], ]),
            
        "attention_mask": torch.tensor( [ [1, 1, 1, 1, 0],
                                        [1, 1, 1, 1, 1], ] ),        
           
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
        german_sentences=["Hallo.", "Guten Morgen."],
        english_sentences=["Hello.", "Good morning."],
        max_length=128,
    )

    assert torch.equal( batch["source_token_ids"], source_encoding["input_ids"], )
    
    assert torch.equal( batch["source_mask"], source_encoding["attention_mask"].bool(), )

    assert torch.equal( batch["decoder_input_ids"], target_encoding["input_ids"][:, :-1], ) 

    assert torch.equal( batch["labels"], target_encoding["input_ids"][:, 1:], )
        
    assert torch.equal( batch["decoder_mask"], target_encoding["attention_mask"][:, :-1].bool(),)
        
    assert ( batch["decoder_input_ids"].shape == batch["decoder_mask"].shape == batch["labels"].shape  )

def test_prepare_translation_batch_rejects_mismatched_sizes() -> None:
    with pytest.raises( ValueError,  match="same number of sentences", ):    
        prepare_translation_batch(
            tokenizer=object(),
            german_sentences=["Hallo.", "Danke."],
            english_sentences=["Hello."],
        )

def test_prepare_translation_batch_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="must not be empty", ):   
        prepare_translation_batch(
            tokenizer=object(),
            german_sentences=[],
            english_sentences=[],
        )