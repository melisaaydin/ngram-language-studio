# NLP Homework 2 – Statistical Language Models

## 1. Project Objective

In this project, I developed a language model that can predict the next word in a sentence and generate text based on a given starting phrase.

I used three statistical language models: Unigram, Bigram, and Trigram. I wanted to see how using different numbers of previous words would affect the predictions and the generated text.

I also created a simple web application using Gradio so that users can try the models, enter their own sentences, and see the results.

## 2. Dataset

For this project, I used the **WikiText-2 English dataset**, which contains text collected from Wikipedia articles.

I chose this dataset because it includes texts on different topics and is commonly used for language modeling tasks.

After preprocessing, the dataset contained:

| Dataset | Number of Sequences |
|---|---:|
| Training | 17,555 |
| Validation | 1,841 |
| Test | 2,183 |

The training data contained 1,718,622 tokens, and the vocabulary included 40,460 words and special tokens.

I used the training set to train the models, the validation set to adjust model parameters, and the test set to evaluate their performance.

## 3. Text Preprocessing

Before training the models, I cleaned and prepared the dataset.

First, I converted the text to lowercase and removed empty lines and Wikipedia headings. Then, I divided the text into individual words using tokenization.

I also checked how frequently each word appeared in the training data. Words that appeared only once were replaced with a special `<UNK>` token, which represents unknown words.

I used this approach to reduce the number of rare words and make the vocabulary easier to manage.

After preprocessing, I saved the data in a JSON file so that I could use it again without repeating the same steps.

## 4. Language Model Implementation

I implemented three different N-gram models to compare their behavior.

- **Unigram:** This model predicts words based only on how frequently they appear in the training data. It does not consider the previous words.
- **Bigram:** This model uses the previous word to predict the next one.
- **Trigram:** This model uses the previous two words, so it can consider more context than the other models.

I calculated word frequencies from the training data and used them to estimate the probability of the next word.

One problem I encountered was that some word combinations did not appear in the training data. To handle this, I used smoothing and backoff techniques. These methods allow the model to use more general word probabilities when there is not enough information about a particular word combination.

I also added a text generation function that selects words based on their probabilities.

## 5. Model Evaluation

To compare the models, I used **perplexity**, which measures how well a language model predicts words in a dataset. A lower perplexity value means that the model makes better predictions on the same evaluation data.

These were my final results:

| Model | Validation PPL | Test PPL |
|---|---:|---:|
| Unigram | 1308.2742 | 1261.8062 |
| Bigram | 1078.2198 | 1029.7699 |
| Trigram | 589.7958 | 573.3467 |

At first, I expected the Trigram model to perform better because it uses more context. However, its initial validation perplexity was 1655.5677, which was worse than the other two models.

I realized that the model was relying too much on word combinations that appeared only a few times in the training data.

To improve this, I tried different values for the `backoff_strength` parameter and compared the results using the validation dataset.

After testing six different values, I found that **500** gave the best result. The Trigram model's validation perplexity decreased to 589.7958.

This was one of the most useful parts of the project because I could clearly see how changing a model parameter affected its performance.

## 6. Web Interface

I used **Gradio** to create a web interface for the project.

The application allows users to enter a starting phrase, choose between the three models, and generate a continuation of the text.

I also added some extra features:

- Selecting how many words to generate.
- Adjusting the temperature to control randomness.
- Viewing the five most probable next words and their probabilities.
- Comparing the perplexity results of the models.

I organized these features into separate tabs to make the application easier to use.

## 7. Results and Observations

When I tested the models through the web interface, I noticed some clear differences.

The **Unigram model** often generated repeated or unrelated words. This made sense because it does not consider the previous words when making predictions.

The **Bigram model** produced slightly more connected phrases, but the sentences were still not always meaningful. It also repeated some word combinations.

The **Trigram model** generally produced more natural short phrases. However, when I generated longer texts, the sentences sometimes changed direction or lost their meaning.

I tried to reduce these problems by adding a repetition penalty and preventing the same three-word combinations from appearing repeatedly.

These changes helped reduce some repetition, but they did not completely solve the problem of generating meaningful sentences.

I understood that this is one of the limitations of N-gram models. Even though the Trigram model uses more context than Unigram and Bigram, it still considers only the previous two words.

## 8. What I Learned

Before starting this project, I understood the basic idea of predicting the next word, but implementing the models helped me understand how the process works in practice.

I learned how to prepare a text dataset, build a vocabulary, calculate N-gram frequencies, and use probabilities to predict words.

I also learned why smoothing is necessary and how perplexity can be used to compare language models.

One of the challenges was improving the Trigram model. Increasing the amount of training data did not automatically give me the results I expected. I had to experiment with the backoff parameter to improve its performance.

Another thing I noticed was that **a lower perplexity value does not always mean that the generated text will sound natural**. Even after improving the Trigram model's perplexity, some generated sentences were still difficult to understand.

Working with Gradio was also useful because I could test my models through a web application instead of only looking at terminal outputs.

Overall, this project helped me understand both how statistical language models work and where their limitations come from.

## 9. Conclusion

In this homework, I developed three N-gram language models using the WikiText-2 dataset and connected them to an interactive Gradio application.

I trained the models, compared their perplexity values, and experimented with different parameters to improve the results.

The Trigram model achieved the best perplexity after tuning, although the generated text was not always meaningful, especially for longer sentences.

The most important thing I learned from this project was that language modeling is not just about choosing the most frequent words. The amount of context, the training data, and the probability estimation method all affect the results.

I think this project was a useful way to understand the basic concepts behind language models before moving on to more advanced neural approaches.