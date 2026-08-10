import random
from collections.abc import Sequence
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from training.tokenization import tokenize_translation_batch


def create_minibatches(
    german_sentences: Sequence[str],
    english_sentences: Sequence[str],
    batch_size: int,
    shuffle: bool = False,
) -> list[tuple[list[str], list[str]]]:
    """
    Split aligned German-English sentence pairs into mini-batches.
    """
    if len(german_sentences) != len(english_sentences):
        raise ValueError( "German and English datasets must contain " "the same number of sentences" )

    if len(german_sentences) == 0:
        raise ValueError("dataset must not be empty" )

    if not isinstance(batch_size, int):
        raise TypeError( "batch_size must be an integer" )

    if batch_size <= 0:
        raise ValueError( "batch_size must be positive" )

    # Create indices so German-English pairs remain aligned.
    indices = list(range(len(german_sentences)))

    if shuffle:
        random.shuffle(indices)

    batches = []

    for start in range( 0, len(indices),batch_size, ):
        batch_indices = indices[ start:start + batch_size ]

        german_batch = [ german_sentences[index] for index in batch_indices ]

        english_batch = [ english_sentences[index] for index in batch_indices ]

        batches.append(( german_batch, english_batch,))

    return batches

def prepare_translation_batch(
    tokenizer: PreTrainedTokenizerBase,
    german_sentences: Sequence[str],
    english_sentences: Sequence[str],
    max_length: int = 128,
) -> dict[str, Tensor]:
    """
    Tokenize a German-English batch and prepare model inputs and labels.
    """
    if len(german_sentences) != len(english_sentences):
        raise ValueError("German and English batches must contain " "the same number of sentences"  )

    if len(german_sentences) == 0:
        raise ValueError( "translation batch must not be empty" )

    source_encoding, target_encoding = (
        tokenize_translation_batch(
            tokenizer=tokenizer,
            german_sentences=german_sentences,
            english_sentences=english_sentences,
            max_length=max_length,
        )
    )

    source_token_ids = source_encoding["input_ids"]
    source_mask = source_encoding["attention_mask"].bool()

    target_token_ids = target_encoding["input_ids"]
    target_mask = target_encoding["attention_mask"].bool()

    if target_token_ids.size(1) < 2:
        raise ValueError( "target sequences must contain at least " "two tokens for target shifting" )

    # Decoder sees every target token except the last one.
    decoder_input_ids = target_token_ids[:, :-1]

    # Labels are shifted one position to the left.
    labels = target_token_ids[:, 1:]

    # The decoder mask must have the same length as decoder_input_ids.
    decoder_mask = target_mask[:, :-1]

    return {
        "source_token_ids": source_token_ids,
        "source_mask": source_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_mask": decoder_mask,
        "labels": labels,
    }