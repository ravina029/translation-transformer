import argparse
import time
import random
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from modelling.transformer import Transformer
from training.batching import (create_minibatches,prepare_translation_batch, move_batch_to_device,)
from training.data import load_translation_pairs
from training.tokenization import ( DEFAULT_MAX_LENGTH, build_tokenizer, )
from training.evaluate import validate_one_epoch


from training.checkpoint import save_checkpoint

#Development configuration
TRAIN_SIZE = 5000
VALIDATION_SIZE = 500
BATCH_SIZE = 4
NUMBER_OF_EPOCHS = 5
HIDDEN_SIZE = 128
NUMBER_OF_HEADS = 4
FEED_FORWARD_SIZE = 256
NUMBER_OF_ENCODER_LAYERS = 2
NUMBER_OF_DECODER_LAYERS = 2
DROPOUT = 0.1

LEARNING_RATE = 1e-4
RANDOM_SEED = 42

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training.
    """
    parser = argparse.ArgumentParser( description="Train a small German-English Transformer on WMT17." )

    parser.add_argument( "--train-size", type=int, default=TRAIN_SIZE,
                        help=f"Number of training sentence pairs (default: {TRAIN_SIZE}).", )

    parser.add_argument( "--validation-size", type=int, default=VALIDATION_SIZE,
                        help=( "Number of validation sentence pairs " f"(default: {VALIDATION_SIZE})." ),)

    parser.add_argument( "--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Mini-batch size (default: {BATCH_SIZE}).", )

    parser.add_argument( "--epochs", type=int, default=NUMBER_OF_EPOCHS,
                        help=f"Number of training epochs (default: {NUMBER_OF_EPOCHS}).", )

    parser.add_argument( "--learning-rate", type=float, default=LEARNING_RATE,
                        help=f"Adam learning rate (default: {LEARNING_RATE}).", )

    return parser.parse_args()

def get_device() -> torch.device:
    """
    Select CUDA, Apple MPS, or CPU depending on availability.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_one_epoch(
    model: Transformer,
    tokenizer: PreTrainedTokenizerBase,
    german_sentences: list[str],
    english_sentences: list[str],
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    ) -> float:
    """
    Training the Transformer for one complete epoch.
    """
    model.train()

    batches = create_minibatches(
        german_sentences=german_sentences,
        english_sentences=english_sentences,
        batch_size=batch_size,
        shuffle=True,
    )

    total_loss = 0.0

    for german_batch, english_batch in batches:
        batch = prepare_translation_batch(
            tokenizer=tokenizer,
            german_sentences=german_batch,
            english_sentences=english_batch,
            max_length=max_length, )

        batch = move_batch_to_device( batch=batch, device=device,)
        optimizer.zero_grad( set_to_none=True )

        logits = model(
            source_token_ids=batch["source_token_ids"],
            target_token_ids=batch["decoder_input_ids"],
            source_mask=batch["source_mask"],
            target_mask=batch["decoder_mask"], )

        loss = criterion( logits.reshape(-1, logits.size(-1), ),
            batch["labels"].reshape(-1), )

        if not torch.isfinite(loss):
            raise ValueError("training loss is not finite")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(batches)


def main() -> None:
    """
    Train and validate a small Transformer on WMT17.
    """
    args = parse_arguments()
    checkpoint_path = ("outputs/checkpoints/" f"transformer_train{args.train_size}_val{args.validation_size}.pt")

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = get_device()
    print( f"Using device: {device}")

    tokenizer = build_tokenizer()
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define a padding token")

    print("Loading training data...")
    german_train, english_train = (load_translation_pairs( split="train", subset_size=args.train_size, ) )

    print("Loading validation data...")
    german_validation, english_validation = (load_translation_pairs( split="validation", subset_size=args.validation_size, ) )

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
        max_length=DEFAULT_MAX_LENGTH,
        dropout=DROPOUT,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, )

    print("\nStarting training")

    training_start_time = time.perf_counter()
    best_validation_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()

        training_loss = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            german_sentences=german_train,
            english_sentences=english_train,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            max_length=DEFAULT_MAX_LENGTH,
        )

        validation_loss = validate_one_epoch(
            model=model,
            tokenizer=tokenizer,
            german_sentences=german_validation,
            english_sentences=english_validation,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            max_length=DEFAULT_MAX_LENGTH,
        )

        epoch_time = time.perf_counter() - epoch_start_time

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Training loss:   {training_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Epoch time:      {epoch_time:.2f} seconds")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                training_loss=training_loss,
                validation_loss=validation_loss,
            )

            print(
                f"Saved new best checkpoint "
                f"with validation loss {validation_loss:.4f}"
            )

    total_time = time.perf_counter() - training_start_time

    print("\nTraining completed.")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Total training time: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()