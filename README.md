# Authorship-attribution-using-ngram-and-LLMs

Overview
This GitHub repository contains an project focused on the authorship attribution problem, which involves determining the author of a given text. The project includes the implementation of both generative (n-gram language model) and discriminative (sequence classifier) solutions. The goal is to write code for these solutions and compare the results to explore the effectiveness of each approach in identifying text authors.

Setup
I have compiled source files containing excerpts from works by various authors: Jane Austen, Charles Dickens, Leo Tolstoy, and Oscar Wilde. These files are available here. Ensure to decide on the encoding type to use, preferably UTF-8.

Project Details
Generative Classifier:

Utilize NLTK's LM package to build n-gram language models, experimenting with different smoothing techniques and backoff strategies as outlined in Section 3.5 of Jurafsky and Martin.
Implement methods to handle out-of-vocabulary words during runtime.
Generate samples for each author using the trained language models and report the perplexity scores.
Optional: Implement n-gram language models from scratch using Numpy or PyTorch for bonus points.
Discriminative Classifier:

Employ Huggingface to create a sequence classification model with k labels corresponding to the number of authors.
Prepare data, create train and test dataloaders, and train the classifier using the Huggingface Trainer class.
Instructions
Run the program classifier.py with the following command-line setups:


python3 classifier.py authorlist -approach [generative|discriminative]
python3 classifier.pt authorlist -approach [generative|discriminative] -test testfile

When running without the -test flag, the program automatically extracts a development set, trains the model, and prints the results.
When running with the -test flag, the program uses the entirety of data to train a language model and outputs classification results for each line in the given test file.
Deliverables

