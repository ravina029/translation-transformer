from collections.abc import Sequence
from transformers import (BatchEncoding, PreTrainedTokenizerBase, AutoTokenizer, )

TOKENIZER_NAME = "facebook/bart-base"
DEFAULT_MAX_LENGTH = 128

def build_tokenizer (local_files_only: bool = False,) -> PreTrainedTokenizerBase:
    """
    Load the shared tokenizer for German and English text without loading the pretrained BART model.
    """
    tokenizer= AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=local_files_only, )

    # in this project the decoder requires padding after the real tokens
    tokenizer.padding_side = "right"

    required_token_ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }

    for name, token_id in required_token_ids.items():
        if token_id is None:
            raise ValueError(f"tokenizer doesn't define {name}")

    # current mask logic requires the beginning and end token to be different from the padding token.
    if tokenizer.bos_token_id == tokenizer.pad_token_id:
        raise ValueError("The BOS and PAD token IDs must be different")

    if tokenizer.eos_token_id == tokenizer.pad_token_id:
        raise ValueError("The EOS and PAD token IDs must be different")

    return tokenizer

def tokenize_sentences(
        tokenizer: PreTrainedTokenizerBase,
        sentences: Sequence[str],
        max_length: int = DEFAULT_MAX_LENGTH, 
        ) -> BatchEncoding:
    """
    Tokenize and dynamically right-pad a batch of sentences.
    """
    if isinstance(sentences, str):
        raise TypeError("sentences must be a sequence of strings, " "not one string")

    if len(sentences) == 0:
        raise ValueError("sentences must contain at least one sentence")

    if not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    for sentence in sentences:
        if not isinstance(sentence, str):
            raise TypeError("Every sentence must be a string")

        if not sentence.strip():
            raise ValueError("Sentences must not be empty")

    return tokenizer(
        list(sentences),
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def tokenize_translation_batch(
    tokenizer: PreTrainedTokenizerBase,
    german_sentences: Sequence[str],
    english_sentences: Sequence[str],
    max_length: int = DEFAULT_MAX_LENGTH,
) -> tuple[BatchEncoding, BatchEncoding]:
    """
    Tokenize one German-English translation batch.
    """

    if len(german_sentences) != len(english_sentences):
        raise ValueError("German and English batches must have the same size")

    source_encoding = tokenize_sentences(
        tokenizer=tokenizer,
        sentences=german_sentences,
        max_length=max_length,
    )

    target_encoding = tokenize_sentences(
        tokenizer=tokenizer,
        sentences=english_sentences,
        max_length=max_length,
    )

    return source_encoding, target_encoding
