# CAST

**CAST**: **C**orpus-**A**ware **S**elf-similarity Enhanced **T**opic Modeling, is a novel topic modeling method that leverages **word embeddings contextualized** on the dataset and **self-similarity scores** to identify contextually meaningful topic words.


<p align="center">
    <img src="https://github.com/amazingmatthew/CAST/blob/main/figs/CAST.png?sanitize=true">
</p>

>Two modules to identify meaningful candidate topic words. Module 1: word embeddings contextualized on the dataset. Module 2: self-similarity scores to filter out functional words. Purple points are documents with semantically similar documents clustered together. Higher-scoring words (green) are served as meaningful candidate topic words (green points) and assigned to their closet clusters.

## Features

- Utilizes pre-trained transformer models for high-quality text embeddings
- Implements word-level and n-gram candidate generation
- Applies UMAP for dimensionality reduction
- Uses HDBSCAN for density-based clustering
- Calculates self-similarity scores for word filtering
- Supports customizable number of topics
- Provides options for verbose output and logging

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- scikit-learn
- UMAP
- HDBSCAN
- pandas
- numpy
- tqdm
- nltk

## Usage

Here's a basic example of how to use the TopicModeling class:

```python
from CAST import CAST

# Prepare your documents
documents = ["Your list of documents here"]
# you can load benchmark datasets such as 20NewsGroups and BBC News from [OCTIS](https://github.com/MIND-Lab/OCTIS)

# Initialize the TopicModeling class
topic_model = CAST(
    documents=documents,
    model_name="bert-base-uncased",  # or any other pre-trained model
    min_count=50,
    self_sim_threshold=0.5,
    nr_topics=10,  # set to None to automatically determine the number of topics
    verbose=True
)

# Run the topic modeling pipeline
topics = topic_model.pipeline()

# Print the resulting topics
for topic_id, words in topics.items():
    print(f"Topic {topic_id}: {', '.join(words)}")
