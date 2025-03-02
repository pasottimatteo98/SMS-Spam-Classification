# Text Mining Project: SMS Spam Classification

This project implements various natural language processing (NLP) techniques to classify SMS messages as spam or ham (not spam). The project uses multiple approaches including traditional machine learning, deep learning with LSTM networks, transfer learning with BERT, and topic modeling.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Data Processing](#data-processing)
- [Models](#models)
- [Topic Modeling](#topic-modeling)
- [Results](#results)
- [Usage](#usage)

## Overview

The project analyzes a dataset of SMS messages to identify spam messages using various text classification techniques. It demonstrates the application of different NLP models for text classification, showcasing the strengths of each approach.

## Project Structure

The project consists of several key components:

- Data preprocessing and cleaning
- Text feature extraction
- Model implementation (LSTM, BERT)
- Topic modeling analysis
- Performance evaluation

## Key Features

- Extensive text preprocessing including stopword removal and abbreviation expansion
- Bidirectional LSTM implementation for sequence classification
- Transfer learning using BERT for text classification
- Topic modeling to identify key themes in the dataset
- Comprehensive model evaluation using precision, recall, and F1 score
- Visualization of results including confusion matrices and word clouds

## Dependencies

```
text_preprocessing
pandas==1.5.3
contextualized_topic_models
pyLDAvis==2.1.2
gensim
transformers
nltk
sklearn
tensorflow
keras
bertopic
wordcloud
matplotlib
seaborn
```

## Data Processing

The project processes SMS data through several steps:

1. Text normalization (lowercase conversion)
2. Special character removal
3. Stopword removal
4. Abbreviation expansion with custom dictionary
5. Text tokenization and padding for neural networks

## Models

### Bidirectional LSTM Network

A custom deep learning model using:
- Embedding layer for text vectorization
- Bidirectional LSTM for sequence processing
- Dense layers with ReLU activation
- Dropout for regularization
- Sigmoid activation for final binary classification

### BERT Transfer Learning

The project leverages pre-trained BERT (bert-base-uncased) for sequence classification:
- Uses BERT tokenizer and preprocessor
- Fine-tunes the model on the SMS spam dataset
- Compares performance with custom LSTM model

## Topic Modeling

The project implements topic modeling using:
- LDA (Latent Dirichlet Allocation) with gensim
- Coherence scoring to determine optimal number of topics
- pyLDAvis visualization of topics
- BERTopic for improved topic modeling

## Results

The project provides various performance metrics:
- Confusion matrices
- Precision, recall, and F1 scores for models
- Topic coherence scores
- Visualizations of spam distribution and topic modeling

## Usage

To use this project:

1. Ensure all dependencies are installed
2. Mount Google Drive to access the dataset
3. Run the preprocessing steps to clean the SMS data
4. Train the models on the processed data
5. Evaluate model performance
6. Analyze topic modeling results

The code includes utility functions for file handling, text preprocessing, and model evaluation.

```python
# Example usage for preprocessing
spam_data = remove_stop_words(spam_data, 'text', 'text')

# Example usage for BERT tokenization
train_inp, train_mask = mask_inputs_for_bert(X_train, max_len)
```

For detailed implementation, refer to the main Python script.
