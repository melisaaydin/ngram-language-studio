
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


# --------------------------------------------------
# 1. Project settings
# --------------------------------------------------

TRAIN_LIMIT = None
VALIDATION_LIMIT = None
TEST_LIMIT = None
MIN_WORD_FREQUENCY = 2

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Download the WikiText-2 dataset
# --------------------------------------------------

print("Downloading WikiText-2 dataset...")

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)

print("\nDataset loaded successfully!")

for split in ["train", "validation", "test"]:
    print(f"{split}: {len(dataset[split]):,} rows")


# --------------------------------------------------
# 3. Text preprocessing
# --------------------------------------------------


def tokenize(text):
    """
    Lowercase text and preserve English contractions.
    """
    text = text.lower()

    # Join apostrophes separated by spaces.
    # Example: island ' s -> island's
    text = re.sub(
        r"(?<=\w)\s*['’]\s*(?=\w)",
        "'",
        text
    )

    return re.findall(
        r"\b\w+(?:'\w+)*\b",
        text,
        flags=re.UNICODE
    )


def preprocess_split(split, limit):
    """
    Remove empty lines and Wikipedia headings,
    then tokenize the remaining text.
    """
    sequences = []

    for text in split["text"]:
        text = text.strip()

        if not text:
            continue

        if text.startswith("="):
            continue

        tokens = tokenize(text)

        if tokens:
            sequences.append(tokens)

        if limit is not None and len(sequences) >= limit:
            break

    return sequences


print("\nPreprocessing text...")

train_sequences = preprocess_split(
    dataset["train"], TRAIN_LIMIT
)

validation_sequences = preprocess_split(
    dataset["validation"], VALIDATION_LIMIT
)

test_sequences = preprocess_split(
    dataset["test"], TEST_LIMIT
)


# --------------------------------------------------
# 4. Build vocabulary using training data only
# --------------------------------------------------

word_counts = Counter(
    word
    for sequence in train_sequences
    for word in sequence
)

vocabulary = {
    word
    for word, count in word_counts.items()
    if count >= MIN_WORD_FREQUENCY
}

vocabulary.add("<UNK>")


# --------------------------------------------------
# 5. Replace unknown words
# --------------------------------------------------

def normalize_sequence(sequence):
    return [
        word if word in vocabulary else "<UNK>"
        for word in sequence
    ]


train_sequences = [
    normalize_sequence(seq) for seq in train_sequences
]

validation_sequences = [
    normalize_sequence(seq) for seq in validation_sequences
]

test_sequences = [
    normalize_sequence(seq) for seq in test_sequences
]


# --------------------------------------------------
# 6. Save prepared data
# --------------------------------------------------

prepared_data = {
    "train": train_sequences,
    "validation": validation_sequences,
    "test": test_sequences,
    "vocabulary": sorted(vocabulary)
}

output_path = OUTPUT_DIR / "prepared_data.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(prepared_data, file, ensure_ascii=False)


# --------------------------------------------------
# 7. Display dataset statistics
# --------------------------------------------------

print("\n--- DATASET STATISTICS ---")

print("Training sequences:", len(train_sequences))
print("Validation sequences:", len(validation_sequences))
print("Test sequences:", len(test_sequences))

print("Vocabulary size:", len(vocabulary))

print(
    "Training tokens:",
    sum(len(seq) for seq in train_sequences)
)

print("\nExample training sequence:")
print(train_sequences[0][:30])

print("\nTop 10 most frequent words:")
for word, count in word_counts.most_common(10):
    print(f"{word}: {count}")

print(f"\nPrepared data saved to: {output_path}")
print("\nDATA PREPARATION COMPLETED SUCCESSFULLY!")
