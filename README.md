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

Here's an example of using CAST to identify topic words on the benchmark 20NewsGroup dataset:

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
```
After this, we are going to print out the topic words for each topic
```python
for topic_id, words in topics.items():
    print(f"Topic {topic_id}: {', '.join(words)}")

>>>
Topic 0: game score team play player season league shot goal tie
Topic 1: patient medical treat treatment medicine doctor disease health hospital sick
Topic 2: secure security encrypt scheme key secret ensure protect privacy clipper
Topic 3: car engine auto vehicle rear dealer motor brake mile gear
Topic 4: monitor vga video card screen display graphic hardware mode slot
Topic 5: launch mission space orbit flight rocket satellite solar moon shuttle
Topic 6: occupy israeli territory arab jewish soldier land peace civilian village
Topic 7: gun firearm weapon shoot armed violent bullet criminal assault batf
Topic 8: gay homosexual homosexuality sexual male sex behavior man adult woman
Topic 9: village armenian turkish soldier army foreign russian soviet occupy organize
```
## Get top_n_sentences associated with the topics
After running the `.pipeline`, we are going to use `.search_docs_by_topic` to get the top_n_sentences associated with the topics. You can specify a `topic_number`, or set `topic_number = None` to get the top_n_sentences for all the topics.

```python
top_sen = model.search_docs_by_topic(topic_number = None, num_docs=10)
```

We are going to print out the topic, topic_size and the top_sentences for each topic
```python
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
We are going to use `.sentiment_analysis_by_topic` to conduct the sentiment analysis for a specific `topic_number`. Setting `topic_number = None` will return sentences for all the clusters. 

```python
sentiment_results = model.sentiment_analysis_by_topic(topic_number=0, num_docs=5) 
```
Returns:
A dictionary where keys are topic numbers, and values are dictionaries containing:
- `top_positive_sentences`: List of top positive sentences with their scores.
- `top_negative_sentences`: List of top negative sentences with their scores.
- `top_neutral_sentences`: List of some neutral sentences examples.
- `avg_sentiment`: Average sentiment score for the topic.
- `positive_ratio`: Ratio of positive sentences.
- `neutral_ratio`: Ratio of neutral sentences.
- `negative_ratio`: Ratio of negative sentences.

We are going to print out the top_positive_sentences and top_negative_sentences for topic 0
```python
for topic, data in sentiment_results.items():
    print(f"\nTopic {topic}:")
    print("-" * 40)
    
    print("\nTop Positive Sentences:")
    for sentence, score in data['top_positive_sentences']:
        print(f"  ('{sentence}', {score})")
    
    print("\nTop Neutral Sentences:")
    for sentence, score in data['top_neutral_sentences']:
        print(f"  ('{sentence}', {score})")
    
    print("\nTop Negative Sentences:")
    for sentence, score in data['top_negative_sentences']:
        print(f"  ('{sentence}', {score})")
    
    print("\nSentiment Statistics:")
    print(f"  Average Sentiment Score: {data['avg_sentiment']}")
    print(f"  Positive Ratio: {data['positive_ratio']}")
    print(f"  Neutral Ratio: {data['neutral_ratio']}")
    print(f"  Negative Ratio: {data['negative_ratio']}")
```

```python
Topic 0:
----------------------------------------

Top Positive Sentences:
  ('happy bob app anybody use happy bob make group', 0.8)
  ('hello hey megan diagnose know diabetes hope maybe meet people personally interesting understand one interested use friend diabete fussy age good', 0.48)
  ('diabetes chat hey sarah people comment love able talk diabete feel free ask question xx n nsent diabetes forum app', 0.47)
  ('look diabetic friend hi ruby type diabete nearly month look talk to ni greatly appreciate like chat friend n nruby x', 0.45)
  ('meet up ect london wonder new kind meet up event london area able find n n n nsent diabetes forum app', 0.41)

Top Neutral Sentences:
  ('think yrs know correct hypo go second', 0.0)
  ('recent twitter meme', 0.0)
  ('today day emoji emoji trialle insight pump week today hopefully like go', 0.0)
  ('diabuddy i m fourteen i ve diabete diagnose wander diabetic like diabetic friend diabuddies n nplease pm', 0.0)
  ('t friend emoji get diagnose month ago know anybody diabete friend group family wonder anybody want diabuddy get insta snapchat xx', 0.0)

Top Negative Sentences:
  ('hiya emoji hi diagnose week ago t d look friend age grateful tip bg test hate', -0.8)
  ('desperate need ero message', -0.6)
  ('friend want hello n ni m type diagnose month bit ago need friend go similar friend understand annoying n nif want touch message email n nabi x', -0.4)
  ('struggle diabetes lately emoji year school mock come try revise experience want test needle anymore tired diabete', -0.35)
  ('type hi dignose type th january ve find difficult come term fact m diabetic constantly feel like outsider m look friend speak diabetes understanding', -0.25)

Sentiment Statistics:
  Average Sentiment Score: 0.09
  Positive Ratio: 0.44
  Neutral Ratio: 0.44
  Negative Ratio: 0.12
```


## Citation



## References
<a name="reference1">1</a>. Xiao, C., Long, Y., & Moubayed, N. A. (2022). On isotropy, contextualization and learning dynamics of contrastive-based sentence representation learning. arXiv preprint arXiv:2212.09170.
