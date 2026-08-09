from collections.abc import Sequence
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from training.tokenization import tokenize_translation_batch


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