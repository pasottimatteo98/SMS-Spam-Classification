# Text Mining Project: SMS Spam Classification

This project involves the classification of SMS messages into 'spam' or 'not spam' categories using various text mining techniques and machine learning models. The project is structured as a Jupyter Notebook.

## Overview

The notebook includes multiple sections, each handling different aspects of the text mining process:

1. Function Definitions: Functions are defined for various tasks including file operations, text preprocessing, tokenization, and visualization.
2. Library and Dataset Import: Necessary libraries are imported and the dataset is loaded.
3. Preprocessing: The dataset undergoes preprocessing which includes renaming columns, removing stopwords, and preparing the text data.
4. Keras Tokenizer: Tokenization of text data and preparation of input sequences for deep learning models.
5. Model Training: Training a Bi-directional LSTM model using Keras.
6. BERT Transfer Learning: Utilizing a pre-trained BERT model for sequence classification.
7. Visualization: Displaying the distribution of spam vs non-spam messages and most frequent words in spam messages.
8. Topic Modeling: Applying LDA to understand the underlying topics in the text data.
9. Evaluation: The models are mainly evaluated based on coherence and perplexity scores.

## File Descriptions

- `SMS Spam dataset.zip`: The dataset used in this project. It needs to be present in Google Drive for the code to access.
- `Text Mining Project Classification.ipynb`: The Jupyter Notebook containing all the code and explanations.

## Instructions

1. Ensure that the dataset is present in your Google Drive.
2. Open the `.ipynb` file in Google Colab or Jupyter Notebook.
3. Run the cells in sequence to avoid dependencies issues.

## Requirements

This project uses several libraries including Pandas, NumPy, NLTK, Scikit-Learn, Gensim, Keras, TensorFlow, Transformers, PyLDAvis, and others. Ensure these are installed before running the notebook.

## Note

- The code contains sections marked with "!", which are shell commands intended to be run in Google Colab. These may need adjustments if run in a different environment.
- The BERT model requires GPU acceleration for efficient training. Make sure to enable GPU in your environment.


