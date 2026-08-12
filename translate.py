import argparse

from modelling.transformer import Transformer
from training.checkpoint import load_checkpoint
from training.inference import greedy_decode
from training.tokenization import build_tokenizer
from training.train import get_device


CHECKPOINT_PATH = ( "outputs/checkpoints/" "transformer_train5000_val500.pt" )

MAX_LENGTH = 128
HIDDEN_SIZE = 128
NUMBER_OF_HEADS = 4
FEED_FORWARD_SIZE = 256
NUMBER_OF_ENCODER_LAYERS = 2
NUMBER_OF_DECODER_LAYERS = 2
DROPOUT = 0.1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for translation.
    """
    parser = argparse.ArgumentParser( description="Translate a German sentence into English." )

    parser.add_argument( "sentence", type=str, help="German sentence to translate.", )

    parser.add_argument( "--checkpoint", type=str, default=CHECKPOINT_PATH,
        help="Path to the trained Transformer checkpoint.", )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = build_tokenizer()

    if tokenizer.pad_token_id is None:
        raise ValueError( "tokenizer must define a padding token" )

    vocabulary_size = len(tokenizer)

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

    metadata = load_checkpoint(
        path=args.checkpoint,
        model=model,
        device=device,
    )

    model.eval()

    print( f"Loaded checkpoint from epoch " f"{metadata['epoch']}" )

    print( f"Validation loss: " f"{metadata['validation_loss']:.4f}" )

    translation = greedy_decode(
        model=model,
        tokenizer=tokenizer,
        german_sentence=args.sentence,
        device=device,
        max_length=MAX_LENGTH,
    )

    print(f"\nGerman:  {args.sentence}")
    print(f"English: {translation}")


if __name__ == "__main__":
    main()
