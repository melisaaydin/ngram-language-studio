
import math

import pytest

from language_model import NGramLanguageModel


# Small dataset for independent unit testing.
# The real WikiText-2 dataset is not required here.
SAMPLE_DATA = [
    ["i", "love", "machine", "learning"],
    ["i", "love", "natural", "language"],
    ["i", "love", "machine", "learning"],
    ["we", "study", "machine", "learning"],
    ["machine", "learning", "is", "interesting"],
]


@pytest.fixture(params=[1, 2, 3])
def trained_model(request):
    model = NGramLanguageModel(n=request.param)
    model.train(SAMPLE_DATA)
    return model


def test_model_is_trained(trained_model):
    assert trained_model.trained is True
    assert trained_model.total_tokens > 0
    assert "</s>" in trained_model.vocabulary


def test_probability_is_valid(trained_model):
    probability = trained_model.probability(
        "machine", ["i", "love"]
    )
    assert 0 < probability <= 1


def test_probabilities_sum_to_one(trained_model):
    context = ["i", "love"]

    total = sum(
        trained_model.probability(word, context)
        for word in trained_model.vocabulary
    )

    assert math.isclose(
        total, 1.0, rel_tol=1e-9, abs_tol=1e-9
    )


def test_unknown_word_handling(trained_model):
    unknown_probability = trained_model.probability(
        "nonexistentword", ["i", "love"]
    )

    unk_probability = trained_model.probability(
        "<UNK>", ["i", "love"]
    )

    assert unknown_probability == unk_probability


def test_predictions_are_sorted(trained_model):
    predictions = trained_model.next_word_predictions(
        "i love", top_k=5
    )

    probabilities = [p for _, p in predictions]

    assert len(predictions) == 5
    assert probabilities == sorted(
        probabilities, reverse=True
    )


def test_text_generation(trained_model):
    continuation = trained_model.generate(
        "i love",
        num_words=10,
        temperature=0.8,
        seed=42
    )

    assert isinstance(continuation, str)
    assert len(continuation.split()) <= 10
    assert "<UNK>" not in continuation
    assert "</s>" not in continuation


def test_perplexity_is_finite(trained_model):
    perplexity = trained_model.perplexity(
        SAMPLE_DATA
    )

    assert math.isfinite(perplexity)
    assert perplexity >= 1.0


def test_save_and_load(trained_model, tmp_path):
    model_path = tmp_path / "test_model.pkl.gz"

    trained_model.save(model_path)

    loaded = NGramLanguageModel.load(model_path)

    original_probability = trained_model.probability(
        "machine", ["i", "love"]
    )

    loaded_probability = loaded.probability(
        "machine", ["i", "love"]
    )

    assert math.isclose(
        original_probability,
        loaded_probability
    )


def test_untrained_model_raises_error():
    model = NGramLanguageModel(n=3)

    with pytest.raises(RuntimeError):
        model.probability("machine", ["i", "love"])


def test_invalid_ngram_order():
    with pytest.raises(ValueError):
        NGramLanguageModel(n=4)


def test_empty_training_data():
    model = NGramLanguageModel(n=3)

    with pytest.raises(ValueError):
        model.train([])


def test_invalid_temperature():
    model = NGramLanguageModel(n=3)
    model.train(SAMPLE_DATA)

    with pytest.raises(ValueError):
        model.generate(
            "i love", temperature=0
        )


def test_generation_is_reproducible():
    model = NGramLanguageModel(n=3)
    model.train(SAMPLE_DATA)

    first = model.generate(
        "i love",
        num_words=20,
        temperature=0.8,
        seed=42
    )

    second = model.generate(
        "i love",
        num_words=20,
        temperature=0.8,
        seed=42
    )

    assert first == second


def test_no_repeated_trigrams():
    model = NGramLanguageModel(n=3)
    model.train(SAMPLE_DATA)

    text = model.generate(
        "i love",
        num_words=20,
        temperature=0.8,
        seed=42
    )

    tokens = model.tokenize("i love") + model.tokenize(text)

    trigrams = [
        tuple(tokens[i:i + 3])
        for i in range(len(tokens) - 2)
    ]

    assert len(trigrams) == len(set(trigrams))
