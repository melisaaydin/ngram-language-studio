
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Load evaluation results
with open(
    "data/training_results.json",
    "r",
    encoding="utf-8"
) as file:
    results = json.load(file)


models = list(results.keys())

validation = [
    results[m]["validation_perplexity"]
    for m in models
]

test = [
    results[m]["test_perplexity"]
    for m in models
]


# Create output directory
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


# Draw comparison chart
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width / 2,
    validation,
    width,
    label="Validation Perplexity"
)

bars2 = ax.bar(
    x + width / 2,
    test,
    width,
    label="Test Perplexity"
)

ax.set_title(
    "N-Gram Language Model Comparison",
    fontsize=15
)

ax.set_ylabel("Perplexity (Lower is Better)")
ax.set_xlabel("Language Model")

ax.set_xticks(x)
ax.set_xticklabels([
    model.capitalize()
    for model in models
])

ax.legend()
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

ax.bar_label(bars1, fmt="%.1f", padding=3)
ax.bar_label(bars2, fmt="%.1f", padding=3)

plt.tight_layout()

output_path = output_dir / "perplexity_comparison.png"

plt.savefig(output_path, dpi=200)
plt.close()

print("Evaluation chart saved:", output_path)
