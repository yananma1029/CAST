# CAST

**CAST**: **C**orpus-**A**ware **S**elf-similarity Enhanced **T**opic Modeling, is a novel topic modeling method that leverages **word embeddings contextualized** on the dataset and **self-similarity scores** to identify contextually meaningful topic words.


<p align="center">
    <img src="https://github.com/amazingmatthew/CAST/blob/main/images/CAST.png?sanitize=true" alt="" width=800 height="whatever">
</p>

>Two modules to identify meaningful candidate topic words. Module 1: word embeddings contextualized on the dataset. Module 2: self-similarity scores to filter out functional words. Purple points are documents with semantically similar documents clustered together. Higher-scoring words (green) are served as meaningful candidate topic words (green points) and assigned to their closet clusters.

### Word embeddings contextualized on the dataset

In this model, word embeddings are encoded based on their **context** within the documents, instead of **statically encoded** in the general domain as standalone tokens. In this case, the embedding for the word "bank" would differ when it appears in the context of a financial institution versus a riverbank.

The word embedding, $E_{\text{word}}$, is given by:

$$E_{\text{word}} = \frac{1}{P} \sum_{k=1}^{P} \text{Em}([x_1, x_2, \ldots, x_n]_k [w])$$

where $P$ is the number of different contexts in which the word appears, and $\text{Em}$ denotes the last hidden state of the embedding model output for the specified word $w$ in the given sentence. The word $w$ embedding was calculated by averaging the subword embeddings (e.g. shame ##less) that form the word.

### Self-similarity score

Inspired by the findings in contrastive learning that functional tokens have fewer self-similarities than that of semantic tokens <sup>[1](#reference1)</sup>, we employ a self-similarity threshold  to effectively filter out functional words or less relevant words and keep meaningful candidate topic words. 


<p align="center">
    <img src="https://github.com/amazingmatthew/CAST/blob/main/images/ablation.png?sanitize=true" alt="" width=600 height="whatever">
</p>

> An optimal range of self-similarity scores can increase topic coherence (TC) and topic diversity (TD), but excessively high thresholds may filter out meaningful words, thereby reducing TC and TD. LLM refers to large language model based metrics.

## Features

- Utilizes pre-trained transformer models for high-quality text embeddings
- Word embeddings contextualized on the dataset
- Implements self-similarity scores to filter out functional and not meaningful words
- Implements word-level and n-gram candidate generation
- Applies UMAP for dimensionality reduction
- Uses HDBSCAN for density-based clustering
- Calculates self-similarity scores for word filtering
- Supports customizable number of topics

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

## Basic Usage

Here's a basic example of how to use CAST using the benchmark 20NewsGroup dataset:

```python

# fetch 20NewsGroup dataset
from CAST import CAST
from sklearn.datasets import fetch_20newsgroups

documents = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

```
Initialize CAST model

```python
topic_model = CAST(
    documents=documents,
    model_name="sentence-transformers/all-mpnet-base-v2", # Support other sentence embedding models
    min_count=50,
    self_sim_threshold=0.3, 
    nr_topics=10,  # set to None to automatically determine the number of topics
    verbose=True
)
```
> The `self_sim_threshold` parameter filters out lower-scoring words, which are typically not meaningful. You can adjust this value (suggested range 0.2-0.5) to achieve optimal results based on your corpus.  An excessively high threshold may filter out meaningful words, leading to lower topic coherence and diversity.

Run the topic modelling pipeline to find topics

```python
topics = topic_model.pipeline()

for topic_id, words in topics.items():
    print(f"Topic {topic_id}: {', '.join(words)}")

>>>
Cluster 0: game, score, team, play, player, season, league, shot, goal, tie
Cluster 1: patient, medical, treat, treatment, medicine, doctor, disease, health, sick, eat
Cluster 2: disk, scsi, ide, controller, device, hardware, bus, rom, tech, boot
Cluster 3: secure, encrypt, security, scheme, key, secret, ensure, protect, privacy, clipper
Cluster 4: launch, mission, space, flight, rocket, orbit, satellite, solar, moon, facility
Cluster 5: doctrine, faith, holy, gospel, church, pray, christian, scripture, verse, god
Cluster 6: occupy, israeli, jewish, arab, territory, soldier, land, peace, civilian, village
Cluster 7: gun, firearm, weapon, armed, shoot, violent, bullet, criminal, assault, batf
Cluster 8: monitor, vga, screen, video, display, card, hardware, tech, graphic, slot
Cluster 9: gay, homosexual, homosexuality, sexual, male, sex, behavior, man, adult, woman
```
Using `.get_top_n_sentences` you can get the top_n_sentences associated with the topics. You can specify a `topic_number`, or set `topic_number = None` to get the top_n_sentences for all the topics.

```python
top_sen = model.get_top_n_sentences(nr_sentences=10, topic_number = None)
print(top_sen)
>>>
   Topic  Count                                      Top_Sentences
0      0     99  [today have not be good, can today be over pls...
1      1     75  [im year old and ve have type diabetes for yea...
2      2     71  [year old type diabetic here can not afford mo...
3      3     62  [a like to be on the go all day so the libre h...
4      4     54  [im a type diabetic ve lose lb and bring my a ...
5      5     44  [once again my blood sugar be somehow low afte...
6      6     33  [i be diagnose with type diabetes year ago tod...
7      7     32  [me a type diabetic or almost year when people...
8      8     27  [thank for host and for the brilliant chat, th...
9      9     27  [hi david m a type diabetic have be for year a...
```


## Citation



## References
<a name="reference1">1</a>. Xiao, C., Long, Y., & Moubayed, N. A. (2022). On isotropy, contextualization and learning dynamics of contrastive-based sentence representation learning. arXiv preprint arXiv:2212.09170.
