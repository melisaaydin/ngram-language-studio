
import gzip
import math
import pickle
import random
import re
from collections import Counter, defaultdict


class NGramLanguageModel:
    """
    Statistical language model supporting
    unigram, bigram and trigram probabilities.
    """

    def __init__(self, n=3, alpha=0.5, backoff_strength=5.0):
        if n not in (1, 2, 3):
            raise ValueError("n must be 1, 2, or 3")

        self.n = n
        self.alpha = alpha
        self.backoff_strength = backoff_strength

        self.counts = {
            order: defaultdict(Counter)
            for order in range(1, n + 1)
        }

        self.vocabulary = set()
        self.total_tokens = 0
        self.trained = False

    def train(self, sequences):
        """Learn n-gram frequencies from training sentences."""

        if not sequences:
            raise ValueError("Training data is empty.")

        for sequence in sequences:
            if not sequence:
                continue

            tokens = list(sequence) + ["</s>"]
            self.vocabulary.update(tokens)

            for order in range(1, self.n + 1):
                padded = ["<s>"] * (order - 1) + tokens

                for i in range(order - 1, len(padded)):
                    context = tuple(
                        padded[i - order + 1:i]
                    ) if order > 1 else ()

                    word = padded[i]
                    self.counts[order][context][word] += 1

        self.vocabulary.add("<UNK>")
        self.total_tokens = sum(
            self.counts[1][()].values()
        )
        
        # Frequent words used as fallback generation candidates.
        self.common_words = [
            word
            for word, _ in self.counts[1][()].most_common(200)
            if word != "<UNK>"
        ]

        self.trained = True

    
    def tokenize(self, text):
        """Tokenize text consistently with preprocessing."""
        text = text.lower()

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


    def normalize(self, tokens):
        """Replace out-of-vocabulary words with UNK."""
        return [
            word if word in self.vocabulary else "<UNK>"
            for word in tokens
        ]

    def probability(self, word, context=()):
        """
        Calculate P(word | context) using
        add-k unigram smoothing and interpolation.
        """

        if not self.trained:
            raise RuntimeError("Model has not been trained.")

        if word not in self.vocabulary:
            word = "<UNK>"

        context = tuple(context)[-(self.n - 1):] \
            if self.n > 1 else ()

        return self._probability_recursive(
            word, context
        )

    def _probability_recursive(self, word, context):
        unigram_counts = self.counts[1][()]
        vocab_size = len(self.vocabulary)

        if not context:
            return (
                unigram_counts[word] + self.alpha
            ) / (
                self.total_tokens + self.alpha * vocab_size
            )

        order = len(context) + 1
        frequency_table = self.counts[order].get(
            context, {}
        )

        context_total = sum(frequency_table.values())

        lower_probability = self._probability_recursive(
            word, context[1:]
        )

        if context_total == 0:
            return lower_probability

        maximum_likelihood = (
            frequency_table.get(word, 0) / context_total
        )

        weight = context_total / (
            context_total + self.backoff_strength
        )

        return (
            weight * maximum_likelihood
            + (1 - weight) * lower_probability
        )

    def next_word_predictions(self, text, top_k=5):
        """Return the highest-probability next words."""

        tokens = self.normalize(self.tokenize(text))

        if self.n > 1:
            context = (
                ["<s>"] * (self.n - 1) + tokens
            )[-(self.n - 1):]
        else:
            context = []

        predictions = [
            (word, self.probability(word, context))
            for word in self.vocabulary
            if word not in {"<UNK>", "</s>"}
        ]

        predictions.sort(
            key=lambda item: (-item[1], item[0])
        )

        return predictions[:top_k]

   
    
    def generate(self, prompt, num_words=20,
                 temperature=1.0, seed=None):
        """
        Generate text with repetition control,
        context-aware sampling and minimum length.
        """

        if not self.trained:
            raise RuntimeError("Model has not been trained.")

        if num_words < 1:
            raise ValueError("num_words must be positive.")

        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        rng = random.Random(seed)

        history = self.normalize(self.tokenize(prompt))
        generated = []

        min_words = min(8, num_words)
        repetition_penalty = 1.4

        common_words = getattr(
            self,
            "common_words",
            [
                word for word, _ in
                self.counts[1][()].most_common(200)
                if word != "<UNK>"
            ]
        )

        for _ in range(num_words):

            if self.n > 1:
                context = (
                    ["<s>"] * (self.n - 1) + history
                )[-(self.n - 1):]
            else:
                context = []

            candidates = set(common_words)

            # Add words observed after matching contexts.
            for order in range(2, self.n + 1):
                sub_context = tuple(
                    context[-(order - 1):]
                )

                table = self.counts[order].get(
                    sub_context, {}
                )

                candidates.update(table.keys())

            candidates.add("</s>")
            candidates.discard("<UNK>")

            # Avoid ending too early.
            if len(generated) < min_words:
                candidates.discard("</s>")

            # Build a list of previously used trigrams.
            previous_trigrams = {
                tuple(history[i:i + 3])
                for i in range(max(0, len(history) - 2))
            }

            scored = []

            for word in candidates:

                # Prevent repeated three-word patterns.
                if len(history) >= 2:
                    new_trigram = tuple(
                        history[-2:] + [word]
                    )

                    if new_trigram in previous_trigrams:
                        continue

                probability = self.probability(
                    word, context
                )

                # Penalize recent word repetition.
                if word in history[-12:] and word != "</s>":
                    probability /= repetition_penalty

                scored.append((word, probability))

            # Relax trigram blocking if no valid candidate remains.
            if not scored:
                scored = [
                    (word, self.probability(word, context))
                    for word in candidates
                ]

            if not scored:
                break

            ranked = sorted(
                scored,
                key=lambda item: (-item[1], item[0])
            )[:40]

            words = [word for word, _ in ranked]

            weights = [
                max(prob, 1e-15) ** (1.0 / temperature)
                for _, prob in ranked
            ]

            selected = rng.choices(
                words,
                weights=weights,
                k=1
            )[0]

            if selected == "</s>":
                break

            generated.append(selected)
            history.append(selected)

        return " ".join(generated)

    def perplexity(self, sequences):
        """
        Evaluate the model on unseen data.
        Lower perplexity generally means better predictions.
        """

        if not sequences:
            raise ValueError("Evaluation data is empty.")

        log_probability_sum = 0.0
        token_count = 0

        for sequence in sequences:
            normalized = self.normalize(sequence)

            history = ["<s>"] * (self.n - 1)

            for word in normalized + ["</s>"]:
                context = history[-(self.n - 1):] \
                    if self.n > 1 else []

                probability = self.probability(
                    word, context
                )

                log_probability_sum += math.log(probability)
                token_count += 1
                history.append(word)

        if token_count == 0:
            raise ValueError("No tokens to evaluate.")

        return math.exp(
            -log_probability_sum / token_count
        )

    def save(self, path):
        """Save the trained model."""
        with gzip.open(path, "wb") as file:
            pickle.dump(
                self,
                file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    @staticmethod
    def load(path):
        """Load a previously saved trusted model."""
        with gzip.open(path, "rb") as file:
            return pickle.load(file)
