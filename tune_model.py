
import json
import shutil
from pathlib import Path

from language_model import NGramLanguageModel


DATA_PATH = Path("data/prepared_data.json")
RESULTS_PATH = Path("data/training_results.json")

MODEL_PATH = Path("models/trigram.pkl.gz")
BACKUP_PATH = Path("models/trigram_original.pkl.gz")

TUNING_RESULTS_PATH = Path("data/tuning_results.json")

# Values to compare on the validation set only.
BACKOFF_VALUES = [5, 20, 100, 500, 2000, 10000]


def main():
    print("=" * 55)
    print("TRIGRAM HYPERPARAMETER TUNING")
    print("=" * 55)

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    validation_data = data["validation"]

    model = NGramLanguageModel.load(MODEL_PATH)

    # Save the original model before any modifications.
    if not BACKUP_PATH.exists():
        shutil.copy2(MODEL_PATH, BACKUP_PATH)

    print("\nLoaded trained Trigram model.")
    print("Vocabulary size:", len(model.vocabulary))
    print("Validation sequences:", len(validation_data))

    tuning_results = []
    best_strength = None
    best_perplexity = float("inf")

    for strength in BACKOFF_VALUES:

        model.backoff_strength = float(strength)

        validation_ppl = model.perplexity(
            validation_data
        )

        tuning_results.append({
            "backoff_strength": strength,
            "validation_perplexity": round(
                validation_ppl, 4
            )
        })

        print(
            f"Backoff = {strength:<6} "
            f"Validation PPL = {validation_ppl:.4f}"
        )

        if validation_ppl < best_perplexity:
            best_perplexity = validation_ppl
            best_strength = strength

    print("\n" + "=" * 55)
    print("BEST CONFIGURATION")
    print("=" * 55)

    print("Best backoff strength:", best_strength)
    print("Best validation PPL:", round(best_perplexity, 4))

    # Select the best setting using validation only.
    model.backoff_strength = float(best_strength)

    # Evaluate test data once after selecting the setting.
    test_perplexity = model.perplexity(data["test"])

    print("Final test PPL:", round(test_perplexity, 4))

    # Save the tuned model for the Gradio application.
    model.save(MODEL_PATH)

    with open(
        TUNING_RESULTS_PATH, "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "experiments": tuning_results,
                "best_backoff_strength": best_strength,
                "best_validation_perplexity": best_perplexity,
                "final_test_perplexity": test_perplexity
            },
            file,
            indent=4
        )

    # Update Gradio's model evaluation table.
    if RESULTS_PATH.exists():
        shutil.copy2(
            RESULTS_PATH,
            Path("data/training_results_before_tuning.json")
        )

        with open(
            RESULTS_PATH, "r", encoding="utf-8"
        ) as file:
            results = json.load(file)

        results["trigram"]["validation_perplexity"] = round(
            best_perplexity, 4
        )

        results["trigram"]["test_perplexity"] = round(
            test_perplexity, 4
        )

        results["trigram"]["backoff_strength"] = best_strength

        with open(
            RESULTS_PATH, "w", encoding="utf-8"
        ) as file:
            json.dump(results, file, indent=4)

    print("\nTuned model saved:", MODEL_PATH)
    print("Tuning results saved:", TUNING_RESULTS_PATH)
    print("\nHYPERPARAMETER TUNING COMPLETED!")


if __name__ == "__main__":
    main()
