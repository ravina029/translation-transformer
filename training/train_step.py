import torch
from torch import nn
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
LEARNING_RATE = 1e-4


def main() -> None:
    """
    Run one complete Transformer training step on a real WMT17 batch.
    """
    tokenizer = build_tokenizer()

    if tokenizer.pad_token_id is None:
        raise ValueError( "tokenizer must define a padding token" )

    german_sentences, english_sentences = (load_translation_pairs( split="train", subset_size=BATCH_SIZE, ) )

    batch = prepare_translation_batch( tokenizer=tokenizer, german_sentences=german_sentences,
                                                            english_sentences=english_sentences, max_length=MAX_LENGTH, )

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

    # Training mode enables dropout.
    model.train()

    criterion = nn.CrossEntropyLoss( ignore_index=tokenizer.pad_token_id, )

    optimizer = torch.optim.Adam( model.parameters(), lr=LEARNING_RATE, )

    # Remove gradients left from any previous optimization step.
    optimizer.zero_grad(set_to_none=True)

    logits = model(
        source_token_ids=batch["source_token_ids"],
        target_token_ids=batch["decoder_input_ids"],
        source_mask=batch["source_mask"],
        target_mask=batch["decoder_mask"],
    )

    labels = batch["labels"]

    print("\nShapes")
    print("logits:", logits.shape)
    print("labels:", labels.shape)

     # CrossEntropyLoss expects:
     # predictions: (number_of_items, vocabulary_size) & labels: (number_of_items) 
     # Therefore batch and sequence dimensions are flattened.

    loss = criterion(
        logits.reshape(
            -1,
            logits.size(-1),
        ),
        labels.reshape(-1),
    )

    if not torch.isfinite(loss):
        raise ValueError( "training loss is not finite" )

    print("\nLoss before backward:")
    print(loss.item())

    # Propagate the loss through the entire Transformer.
    loss.backward()

    output_gradient = model.output_projection.weight.grad

    if output_gradient is None:
        raise ValueError( "output projection did not receive gradients" )

    if not torch.isfinite(output_gradient).all():
        raise ValueError( "output projection contains non-finite gradients" )

    gradient_norm = output_gradient.norm().item()

    print("\nOutput projection gradient norm:")
    print(gradient_norm)

    if gradient_norm == 0.0:
        raise ValueError( "output projection gradient is zero" )

    parameter_before = ( model.output_projection.weight[0].detach().clone() )

     # Update model parameters using the gradients.
    optimizer.step()

    parameter_after = ( model.output_projection.weight[0].detach().clone() )

    parameter_change = ( parameter_after - parameter_before ).norm().item()

    print("\nParameter change after optimizer step:")
    print(parameter_change)
    
    if parameter_change == 0.0:
        raise ValueError(
            "optimizer did not update model parameters"
        )
    
    print("\nTraining step passed.")
    print(
        "Loss, backward propagation, and parameter "
        "update completed successfully."
    )


if __name__ == "__main__":
    main()



