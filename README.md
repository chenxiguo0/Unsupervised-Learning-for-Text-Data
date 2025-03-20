[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/TduUs4Bn)
# Unsupervised Learning for Text Data

## Overview

Exploratory Data Analysis (EDA) has been a fundamental concept in data science since John Tukey’s 1977 book *Exploratory Data Analysis*. However, traditional EDA methods primarily focus on numeric and tabular data, while textual data presents unique challenges and opportunities. This project extends EDA techniques to text data by leveraging two modern unsupervised learning approaches:

- **Topic Modeling**: Extracting latent themes within a text corpus.
- **Word Embeddings**: Capturing semantic relationships between words based on contextual usage.

Through these methods, we explore how unsupervised learning can provide meaningful insights into large-scale text datasets.

## Topic Modeling for Corpus-Level Analysis

The project begins with an investigation of a corpus of **1,529 New York Times articles** published in January 2007 across five sections:

| Section        |
|---------------|
| US News      |
| World News   |
| Arts         |
| Sports       |
| Real Estate  |

Using **Topic Modeling**, we assess whether an unsupervised approach can uncover the original article categories or propose alternative thematic structures. This exercise serves as both an introduction to topic modeling and a calibration step for understanding unsupervised clustering methods.

## Word Embeddings for Token-Level Analysis

Building upon topic modeling, the project explores **Word Embeddings**, a neural NLP technique that maps words into a high-dimensional space based on their contextual usage. Unlike statistical topic models, which explicitly define word distributions, word embeddings implicitly learn linguistic relationships by predicting missing words in text.

By analyzing pre-trained word vectors derived from the same **New York Times** corpus, we examine:
- Semantic relationships between words
- Clustering of words based on contextual similarity
- The impact of dimensionality reduction on interpretability

## Implementation

The project is structured into four parts, implemented in Jupyter notebooks:

| Part | Methodology       | Notebook                 |
|------|-----------------|--------------------------|
| 1    | Topic Modeling  | `part-1.ipynb`  |
| 2    | Topic Modeling  | `part-2.ipynb`  |
| 3    | Word Embeddings | `part-3.ipynb`  |
| 4    | Word Embeddings | `part-4.ipynb`  |

Each notebook provides step-by-step instructions for replicating the analysis. The code and results are documented within the repository to facilitate exploration and extension of the methods.

## Conclusion

By applying unsupervised learning techniques to text data, this project bridges the gap between traditional EDA and modern NLP methodologies. The insights gained from topic modeling and word embeddings can serve as a foundation for further exploration in text mining, clustering, and representation learning.
