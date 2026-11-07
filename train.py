
import json
import time
from pathlib import Path

from language_model import NGramLanguageModel


DATA_PATH = Path("data/prepared_data.json")
MODEL_DIR = Path("models")
RESULTS_PATH = Path("data/training_results.json")

MODEL_DIR.mkdir(exist_ok=True)


def load_prepared_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Prepared dataset not found. "
            "Please run prepare_data.py first."
        )

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("=" * 50)
    print("NLP HOMEWORK 2 - N-GRAM MODEL TRAINING")
    print("=" * 50)

    data = load_prepared_data()

    train_data = data["train"]
    validation_data = data["validation"]
    test_data = data["test"]

    results = {}

    models = {
        "unigram": 1,
        "bigram": 2,
        "trigram": 3
    }

    for model_name, n in models.items():
        print(f"\nTraining {model_name.upper()} model...")

        start_time = time.time()

        model = NGramLanguageModel(
            n=n,
            alpha=0.5,
            backoff_strength=5.0
        )

        model.train(train_data)

        training_time = time.time() - start_time

        print("Training completed.")
        print(f"Training time: {training_time:.2f} seconds")

        print("Calculating validation perplexity...")
        validation_perplexity = model.perplexity(
            validation_data
        )

        print("Calculating test perplexity...")
        test_perplexity = model.perplexity(
            test_data
        )

        model_path = MODEL_DIR / f"{model_name}.pkl.gz"
        model.save(model_path)

        results[model_name] = {
            "n": n,
            "training_time_seconds": round(
                training_time, 3
            ),
            "vocabulary_size": len(model.vocabulary),
            "validation_perplexity": round(
                validation_perplexity, 4
            ),
            "test_perplexity": round(
                test_perplexity, 4
            ),
            "model_path": str(model_path)
        }

        print(f"Vocabulary size: {len(model.vocabulary)}")
        print(f"Validation perplexity: {validation_perplexity:.4f}")
        print(f"Test perplexity: {test_perplexity:.4f}")
        print(f"Saved: {model_path}")

        print("\nExample next-word predictions:")
        predictions = model.next_word_predictions(
            "the history of", top_k=5
        )

        for word, probability in predictions:
            print(f"  {word}: {probability:.5f}")

    with open(
        RESULTS_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(
        f"{'Model':<12} {'Validation PPL':<18} {'Test PPL':<12}"
    )

    for model_name, metrics in results.items():
        print(
            f"{model_name:<12} "
            f"{metrics['validation_perplexity']:<18.4f} "
            f"{metrics['test_perplexity']:<12.4f}"
        )

    best_model = min(
        results,
        key=lambda name: results[name]["validation_perplexity"]
    )

    print(f"\nBest validation model: {best_model.upper()}")

    print("\nALL MODELS TRAINED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
