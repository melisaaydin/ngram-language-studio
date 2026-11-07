
from pathlib import Path
import json

import gradio as gr
import pandas as pd

from language_model import NGramLanguageModel


# --------------------------------------------------
# 1. Load trained models
# --------------------------------------------------

MODEL_DIR = Path("models")

MODEL_FILES = {
    "Unigram": MODEL_DIR / "unigram.pkl.gz",
    "Bigram": MODEL_DIR / "bigram.pkl.gz",
    "Trigram": MODEL_DIR / "trigram.pkl.gz"
}


def load_models():
    models = {}

    for name, path in MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Model file missing: {path}\n"
                "Please run train.py first."
            )

        models[name] = NGramLanguageModel.load(path)
        print(f"Loaded {name}: {path}")

    return models


MODELS = load_models()


# --------------------------------------------------
# 2. Model statistics
# --------------------------------------------------

RESULTS_PATH = Path("data/training_results.json")


def load_results():
    if not RESULTS_PATH.exists():
        return pd.DataFrame()

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []

    for name, metrics in results.items():
        rows.append({
            "Model": name.capitalize(),
            "Vocabulary": metrics["vocabulary_size"],
            "Validation PPL": metrics["validation_perplexity"],
            "Test PPL": metrics["test_perplexity"]
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 3. Predict next words
# --------------------------------------------------

def predict_words(prompt, model_name):
    if not prompt or not prompt.strip():
        return pd.DataFrame(
            columns=["Word", "Probability (%)"]
        )

    model = MODELS[model_name]

    predictions = model.next_word_predictions(
        prompt,
        top_k=5
    )

    return pd.DataFrame([
        {
            "Word": word,
            "Probability (%)": round(prob * 100, 3)
        }
        for word, prob in predictions
    ])


# --------------------------------------------------
# 4. Generate text
# --------------------------------------------------

def generate_text(prompt, model_name, num_words,
                  temperature):

    if not prompt or not prompt.strip():
        return "", "Please enter a starting phrase."

    model = MODELS[model_name]

    continuation = model.generate(
        prompt=prompt,
        num_words=int(num_words),
        temperature=float(temperature)
    )

    if not continuation:
        return "", (
            "The model predicted the end of the sentence. "
            "Try another phrase."
        )

    full_text = prompt.strip() + " " + continuation

    return continuation, full_text


# --------------------------------------------------
# 5. Gradio web interface
# --------------------------------------------------

with gr.Blocks(
    title="N-Gram Language Studio"
) as demo:

    gr.Markdown(
        """
        # N-Gram Language Studio

        ### Statistical Language Modeling · Explore, Predict & Generate

        Generate English text and explore word probabilities
        using models trained on the WikiText-2 dataset.

        **Models:** Unigram · Bigram · Trigram
        """
    )

    with gr.Tabs():

        with gr.Tab("Text Generation"):

            with gr.Row():

                with gr.Column():

                    model_choice = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value="Trigram",
                        label="Choose Language Model"
                    )

                    prompt_input = gr.Textbox(
                        label="Starting Phrase",
                        value="the history of",
                        placeholder="Enter an English phrase...",
                        lines=3
                    )

                    word_slider = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Number of Words"
                    )

                    temperature_slider = gr.Slider(
                        minimum=0.2,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="Temperature"
                    )

                    generate_button = gr.Button(
                        "Generate Text",
                        variant="primary"
                    )

                with gr.Column():

                    continuation_output = gr.Textbox(
                        label="Generated Continuation",
                        lines=5,
                        interactive=False
                    )

                    full_text_output = gr.Textbox(
                        label="Complete Text",
                        lines=7,
                        interactive=False
                    )

            generate_button.click(
                fn=generate_text,
                inputs=[
                    prompt_input,
                    model_choice,
                    word_slider,
                    temperature_slider
                ],
                outputs=[
                    continuation_output,
                    full_text_output
                ]
            )

            gr.Examples(
                examples=[
                    ["the history of"],
                    ["the world is"],
                    ["in the early years"],
                    ["the development of"],
                    ["one of the most important"]
                ],
                inputs=[prompt_input]
            )

        with gr.Tab("Next Word Prediction"):

            gr.Markdown(
                "### Discover the Most Probable Next Words"
            )

            prediction_prompt = gr.Textbox(
                label="Enter a Phrase",
                value="the history of",
                lines=2
            )

            prediction_model = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Trigram",
                label="Language Model"
            )

            predict_button = gr.Button(
                "Predict Next Words",
                variant="primary"
            )

            prediction_table = gr.Dataframe(
                headers=["Word", "Probability (%)"],
                label="Top 5 Predictions",
                interactive=False
            )

            predict_button.click(
                fn=predict_words,
                inputs=[
                    prediction_prompt,
                    prediction_model
                ],
                outputs=prediction_table
            )

        with gr.Tab("Model Evaluation"):

            gr.Markdown(
                """
                ### Model Performance Comparison

                **Perplexity (PPL)** measures how well
                a language model predicts unseen text.

                Lower perplexity generally indicates
                better predictive performance on the
                same evaluation dataset.
                """
            )

            results_table = gr.Dataframe(
                value=load_results(),
                interactive=False,
                label="Model Evaluation Results"
            )

        with gr.Tab("About"):

            gr.Markdown(
                """
                ## About This Project

                An interactive project for exploring
                statistical word prediction.

                **Dataset:** WikiText-2 (English)

                **Approach:** Statistical N-gram Language Models

                **Implemented Models:**
                - Unigram
                - Bigram
                - Trigram

                **Features:**
                - Text preprocessing
                - N-gram frequency estimation
                - Add-k unigram smoothing
                - Interpolated backoff
                - Next-word prediction
                - Text generation
                - Perplexity-based evaluation
                - Interactive Gradio interface

                This project implements statistical language
                modeling without using pretrained LLMs.
                """
            )


# --------------------------------------------------
# 6. Launch application
# --------------------------------------------------

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False
    )
