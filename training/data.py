from datasets import load_dataset

DATASET_NAME= "wmt/wmt17"
DATASET_CONFIG= "de-en"

TRAIN_SIZE= 1000
VALIDATION_SIZE= 200
TEST_SIZE= 200

VALID_SPLITS= {"train", "validation", "test"}

def preprocess_sentence(sentence:str) -> str:
    """Apply minimal whitespace cleaning to one sentence"""

    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")

    cleaned_sentence=" ".join(sentence.strip().split())

    if not cleaned_sentence:
        raise ValueError("sentence must not be empty")

    return cleaned_sentence

def preprocess_translation_pair(
        german_sentence : str,
        english_sentence : str, 
        ) -> tuple[str, str]:
    """
    validate and clean one German-Enlish sentence Pair.
    """
    german_sentence = preprocess_sentence(
        german_sentence
    )

    english_sentence = preprocess_sentence(
        english_sentence
    )

    return german_sentence, english_sentence


def load_translation_pairs(
        split: str,
        subset_size: int,
        ) -> tuple[list[str],list[str]]:

    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, " f"got {split!r}")

    if not isinstance(subset_size, int):
        raise TypeError( "subset_size must be an integer" )

    if subset_size <= 0:
        raise ValueError( "subset_size must be positive" )

    dataset= load_dataset(DATASET_NAME,
                          DATASET_CONFIG,
                          split= split,
                          streaming=True, )

    german_sentences=[]
    english_sentences=[]

    for example in dataset:
        translation= example.get("translation")

        if not isinstance(translation, dict):
            continue

        german_sentence = translation.get("de")
        english_sentence = translation.get("en")

        try:
            german_sentence, english_sentence = (
                preprocess_translation_pair(
                    german_sentence,
                    english_sentence,
                )
            )

        except (TypeError, ValueError):
            # Skip malformed or empty sentence pairs.
            continue

        german_sentences.append(german_sentence)
        english_sentences.append(english_sentence)

        if len(german_sentences) == subset_size:
            break

    if len(german_sentences) != subset_size:
        raise ValueError( f"Requested {subset_size} valid examples from "
            f"{split!r}, but found only " f"{len(german_sentences)}" )

    return german_sentences, english_sentences




    

 

