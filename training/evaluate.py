import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from modelling.transformer import Transformer
from training.batching import (
    create_minibatches,
    move_batch_to_device,
    prepare_translation_batch,
)

def validate_one_epoch(
    model: Transformer,
    tokenizer: PreTrainedTokenizerBase,
    german_sentences: list[str],
    english_sentences: list[str],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    batch_size: int,
    max_length: int, ) -> float:
    """
    Evaluate the Transformer on the validation subset.
    """
    model.eval()

    batches = create_minibatches(
        german_sentences=german_sentences,
        english_sentences=english_sentences,
        batch_size=batch_size,
        shuffle=False, )

    total_loss = 0.0
    with torch.no_grad():
        for german_batch, english_batch in batches:

            batch = prepare_translation_batch(
                tokenizer=tokenizer,
                german_sentences=german_batch,
                english_sentences=english_batch,
                max_length=max_length,
            )

            batch = move_batch_to_device( batch=batch, device=device,)
            logits = model(
                        source_token_ids=batch[ "source_token_ids" ],
                        target_token_ids=batch[ "decoder_input_ids" ],
                        source_mask=batch[ "source_mask" ],
                        target_mask=batch[ "decoder_mask" ], )

            loss = criterion( logits.reshape(-1, logits.size(-1), ),
                        batch["labels"].reshape(-1), )

            if not torch.isfinite(loss):
                        raise ValueError( "validation loss is not finite" )

            total_loss += loss.item()

    return total_loss / len(batches)
