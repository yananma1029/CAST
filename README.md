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
    self_sim_threshold=0.5, 
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
Cluster 0: game score team play player season league shot goal tie
Cluster 1: patient medical treat treatment medicine doctor disease health hospital sick
Cluster 2: secure security encrypt scheme key secret ensure protect privacy clipper
Cluster 3: car engine auto vehicle rear dealer motor brake mile gear
Cluster 4: monitor vga video card screen display graphic hardware mode slot
Cluster 5: launch mission space orbit flight rocket satellite solar moon shuttle
Cluster 6: occupy israeli territory arab jewish soldier land peace civilian village
Cluster 7: gun firearm weapon shoot armed violent bullet criminal assault batf
Cluster 8: gay homosexual homosexuality sexual male sex behavior man adult woman
Cluster 9: village armenian turkish soldier army foreign russian soviet occupy organize
```
## Get top_n_sentences associated with the topics
After running the `.pipeline` you can use `.search_docs_by_topic` to get the top_n_sentences associated with the topics. You can specify a `topic_number`, or set `topic_number = None` to get the top_n_sentences for all the topics.

```python
top_sen = model.search_docs_by_topic(topic_number = None, num_docs=10)
print(top_sen)
>>>
   Topic  Count                                      Top_Sentences
0      0   1476  [straight game score late run run yesterday pi...
1      1    459  [provide advice concern follow health problem ...
2      2    440  [follow document clipper chip programming chip...
3      3    397  [buy full size opinion vehicle significant pro...
4      4    357  [request follow couple week make mind buy mach...
5      5    294  [time step back research space approach shuttl...
6      6    261  [news miss medium israeli soldier israeli sold...
7      7    247  [survey show public type gun control acceptabl...
8      8    207  [recent study show number man engage homosexua...
9      9    183  [convince sense world cold participate turkish...
```

## Sentiment analysis
You can use `.sentiment_analysis_by_topic` to conduct the sentiment analysis for a specific `topic_number`. Setting `topic_number = None` will return sentences for all the clusters. We use social media data instead of the 20NewsGroup to demonstrate this function because the sentiment expressed on social media tends to be more varied and subjective, whereas news articles are generally more objective.

```python
positive_sen, negtive_sen = model.sentiment_analysis_by_topic(topic_number=1, num_docs=5) 
```
Returns:
- `top_positive_sentences`: A dictionary where keys are topic numbers and values are lists of tuples with top positive sentences and their sentiment scores.
- `top_negative_sentences`: A dictionary where keys are topic numbers and values are lists of tuples with top negative sentences and their sentiment scores.

Print out the results:
```python
print('Positive:')
for topic, sens in positive_sen.items():
    print(f" Topic: {topic}")
    for sen in sens:
        print (f" {sen}")

print('Negative:')
for topic, sen in negtive_sen.items():
    print(f"Topic: {topic}")
    for sen in sens:
        print (f" {sen}")
```

```python
Positive:
 Topic: 1
 ('excellent news rob m a type diabetic and m hope to get my vaccine around marchapril time', 1.0)
 ('good be get a rd shot soon bc have type diabetes amp addison s disease and despite be an athletic teenager my immune system be like a cranky old man refuse to get up out of his recliner and do its part', 0.4)
 ('beside live with type diabete for almost year be a healthy yearold college student so be beyond bless to receive the vaccine read more about maguire student blogger s experience receive her covid vaccine here', 0.38)
 ('tripleboosted wife be a covid nurse we be mask consistently have and so far so good no covid', 0.35)
 ('i have a mild case in june but call my dr s office for paxlovid bc have addison s amp type diabetes be tell qualify bc of those condition but bc of they would need to go to the er to get the med despite be contagious amp there be a hour wait in the waiting room', 0.33)

Negative:
Topic: 1
 ('excellent news rob m a type diabetic and m hope to get my vaccine around marchapril time', 1.0)
 ('good be get a rd shot soon bc have type diabetes amp addison s disease and despite be an athletic teenager my immune system be like a cranky old man refuse to get up out of his recliner and do its part', 0.4)
 ('beside live with type diabete for almost year be a healthy yearold college student so be beyond bless to receive the vaccine read more about maguire student blogger s experience receive her covid vaccine here', 0.38)
 ('tripleboosted wife be a covid nurse we be mask consistently have and so far so good no covid', 0.35)
 ('i have a mild case in june but call my dr s office for paxlovid bc have addison s amp type diabetes be tell qualify bc of those condition but bc of they would need to go to the er to get the med despite be contagious amp there be a hour wait in the waiting room', 0.33)
```


## Citation



## References
<a name="reference1">1</a>. Xiao, C., Long, Y., & Moubayed, N. A. (2022). On isotropy, contextualization and learning dynamics of contrastive-based sentence representation learning. arXiv preprint arXiv:2212.09170.
