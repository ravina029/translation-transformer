import torch
from modelling.transformer import Transformer
from training.batching import prepare_translation_batch
from training.data import load_translation_pairs
from training.tokenization import build_tokenizer


BATCH_SIZE = 4
MAX_LENGTH = 128

HIDDEN_SIZE = 128
NUMBER_OF_HEADS = 4
FEED_FORWARD_SIZE = 256
NUMBER_OF_ENCODER_LAYERS = 2
NUMBER_OF_DECODER_LAYERS = 2
DROPOUT = 0.1

def main() -> None:
    """
    Run one real WMT17 batch through the complete Transformer.
    """
    tokenizer = build_tokenizer()

    # Load a very small real WMT17 batch.
    german_sentences, english_sentences = (
        load_translation_pairs(split="train", subset_size=BATCH_SIZE, )
    )

    # Convert raw sentences into model-ready tensors.
    batch = prepare_translation_batch(
        tokenizer=tokenizer, 
        german_sentences=german_sentences,
        english_sentences=english_sentences, 
        max_length=MAX_LENGTH, )

    # same tokenizer is used for source and target languages.
    vocabulary_size = len(tokenizer)

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define a padding token")

    # Using a small Transformer for the smoke test.
    model = Transformer(
        source_vocabulary_size=vocabulary_size,
        target_vocabulary_size=vocabulary_size,
        hidden_size=HIDDEN_SIZE,
        number_of_heads=NUMBER_OF_HEADS,
        feed_forward_size=FEED_FORWARD_SIZE,
        number_of_encoder_layers=NUMBER_OF_ENCODER_LAYERS,
        number_of_decoder_layers=NUMBER_OF_DECODER_LAYERS,
        source_padding_token_id=tokenizer.pad_token_id,
        target_padding_token_id=tokenizer.pad_token_id,
        max_length=MAX_LENGTH,
        dropout=DROPOUT,
    )

    # Evaluation mode disables dropout randomness.
    model.eval()

    # No gradients are needed for this smoke test.
    with torch.no_grad():
        logits = model(
            source_token_ids=batch["source_token_ids"],
            target_token_ids=batch["decoder_input_ids"],
            source_mask=batch["source_mask"],
            target_mask=batch["decoder_mask"],)

    print("\nInput shapes")
    print("source_token_ids:", batch["source_token_ids"].shape, )
    print("decoder_input_ids:", batch["decoder_input_ids"].shape, )
    print("labels:", batch["labels"].shape,)
    print("\nModel output")
    print("logits:", logits.shape)   

    expected_shape = (
        BATCH_SIZE,
        batch["decoder_input_ids"].size(1),
        vocabulary_size,
    )

    assert logits.shape == expected_shape
    assert torch.isfinite(logits).all()

    print("\nSmoke test passed.")
    print("Real WMT17 data successfully passed " "through the Transformer.") 

if __name__ == "__main__":
    main()
