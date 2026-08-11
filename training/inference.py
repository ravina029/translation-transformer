import torch
from transformers import PreTrainedTokenizerBase
from modelling.transformer import Transformer


def greedy_decode(
    model: Transformer,
    tokenizer: PreTrainedTokenizerBase,
    german_sentence: str,
    device: torch.device,
    max_length: int = 128,
) -> str:
    """
    Translate one German sentence using greedy autoregressive decoding.
    """
    if not isinstance(german_sentence, str):
        raise TypeError( "german_sentence must be a string" )

    if not german_sentence.strip():
        raise ValueError( "german_sentence must not be empty" )

    if tokenizer.bos_token_id is None:
        raise ValueError( "tokenizer must define a BOS token" )

    if tokenizer.eos_token_id is None:
        raise ValueError( "tokenizer must define an EOS token" )

    if max_length <= 1:
        raise ValueError( "max_length must be greater than 1" )

    model.eval()

    # Tokenize the German source sentence.
    source_encoding = tokenizer(
        german_sentence,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    source_token_ids = source_encoding[ "input_ids" ].to(device)

    source_mask = source_encoding["attention_mask" ].bool().to(device)

    # Decoder starts only with BOS.
    generated_token_ids = torch.tensor( [[tokenizer.bos_token_id]], dtype=torch.long, device=device, )

    with torch.no_grad():
        for _ in range(max_length - 1):
            target_mask = torch.ones_like( generated_token_ids, dtype=torch.bool, )

            logits = model(
                source_token_ids=source_token_ids,
                target_token_ids=generated_token_ids,
                source_mask=source_mask,
                target_mask=target_mask, )

            # Only the final decoder position predicts the next token.
            next_token_logits = logits[:, -1, :]

            next_token_id = torch.argmax( next_token_logits, dim=-1, keepdim=True, )

            generated_token_ids = torch.cat(
                [ generated_token_ids,
                    next_token_id, ],
                dim=1,
            )

            # Stop once EOS is generated.
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    translation = tokenizer.decode( generated_token_ids[0], skip_special_tokens=True, )

    return translation.strip()
