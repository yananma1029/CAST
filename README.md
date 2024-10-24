# CAST

**CAST**: **C**orpus-**A**ware **S**elf-similarity Enhanced **T**opic Modeling, is a novel topic modeling method that leverages **word embeddings contextualized** on the dataset and **self-similarity scores** to identify contextually meaningful topic words.


<p align="center">
    <img src="https://github.com/amazingmatthew/CAST/blob/main/images/CAST.png?sanitize=true" alt="" width=800 height="whatever">
</p>

>Two modules to identify meaningful candidate topic words. Module 1: word embeddings contextualized on the dataset. Module 2: self-similarity scores to filter out functional words. Purple points represent documents, with semantically similar ones clustered together. Words with higher self-similarity scores (green) are selected over those with lower scores and assigned to their closest cluster centroid (topic vector: yellow point) as topic words (green points), rather than relying on general-domain topic words (red points).


### Word embeddings contextualized on the dataset

In this model, word embeddings are encoded based on their **context** within the documents, instead of **statically encoded** in the general domain as standalone tokens. In this case, the embedding for the word *"driver"* would be different in the computer science domain compared with the general domain.

The word embedding, $E_{\text{final}}$, is given by:

$$E_{\text{final}} = \frac{1}{P} \sum_{k=1}^{P} \text{Em}([x_1, x_2, \ldots, x_n]_k [Vw])$$

where $P$ is the number of different contexts in which the word appears, and $\text{Em}$ denotes the last hidden state of the embedding model output for the specified word $w$ in the given sentence. The word embedding $Vw$ was calculated by averaging the sub-word embeddings (e.g. shame ##less) that form the word.


### Self-similarity score

Inspired by the findings in contrastive learning that functional tokens have fewer self-similarities than that of semantic tokens <sup>[1](#reference1)</sup>, we employ a self-similarity threshold  to effectively filter out functional words or less relevant words and keep meaningful candidate topic words.

<p align="center">
    <img src="https://github.com/amazingmatthew/CAST/blob/main/images/ablation.png?sanitize=true" alt="" width=600 height="whatever">
</p>

> Above is the ablation study of self-similarity scores. An optimal range of self-similarity scores can increase topic coherence (TC) and topic diversity (TD), but excessively high thresholds may filter out meaningful words, thereby reducing TC and TD. LLM refers to large language model based metrics.
-->
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
    self_sim_threshold=0, # Threshold to filter out lowering score words, typically functional words 
    nr_topics=10,  # set to None to automatically determine the number of topics
    verbose=True
)
```
### Determine an appropriate self-similarity threshold

Set `self_sim_threshold = 0` (no filtering) and sort the `self-similarity scores` to manually find a threshold which can filter out most of the functional words while retaining the meaningful ones.

```python
topics, ss_scores = topic_model.pipeline()
```
```python
sorted_ss_score = sorted(ss_score.items(), key=lambda item: item[1], reverse=True)
high_ss = sorted_ss_score[:10]
upper_ss = [item for item in sorted_ss_score if item[1] < 0.5][:10] # you can adjust the threshold and number of words to display 
low_ss = [item for item in sorted_ss_score if item[1] < 0.4][:10]
lowest_ss = [item for item in sorted_ss_score if item[1] < 0.3][:10]

print(f"high_ss: {high_ss}")
print(f"upper_ss: {upper_ss}")
print(f"low_ss: {low_ss}")
print(f"lowest_ss: {lowest_ss}") 
```
```python
>>>
high_ss: [('armenian', 0.833), ('genocide', 0.781), ('turkish', 0.772), ('homosexuality', 0.764), ('atheism', 0.755), ('arab', 0.753), ('encryption', 0.738), ('massacre', 0.737), ('homosexual', 0.735), ('israeli', 0.732)]
upper_ss: [('student', 0.499), ('heart', 0.499), ('score', 0.499), ('split', 0.499), ('motor', 0.499), ('legitimate', 0.499), ('traditional', 0.499), ('fact', 0.498), ('sequence', 0.498), ('software', 0.498)]
low_ss: [('capability', 0.399), ('driver', 0.399), ('operation', 0.399), ('medium', 0.399), ('practice', 0.399), ('confirm', 0.399), ('exit', 0.399), ('call', 0.398), ('error', 0.398), ('default', 0.398)]
lowest_ss: [('<pad>', 0.299), ('great', 0.299), ('stay', 0.298), ('setting', 0.298), ('good', 0.297), ('highly', 0.297), ('possibly', 0.296), ('extremely', 0.295), ('due', 0.294), ('advance', 0.293)]
```
In this case, 0.4 could achieve the balance. Then, we update the threshold and re-run the model to identify topics.

### Identify topic words

```python
topics, _ = topic_model.pipeline()
```
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
top_sen = topic_model.search_docs_by_topic(topic_number = None, num_docs=10)
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
CAST uses the [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) model to conduct sentiment analysis.

```python
sentiment_results = topic_model.sentiment_analysis_by_topic(topic_number=0, num_docs=5) 
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
    print(f"Topic {topic}:")
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

>>>
Topic 0:
----------------------------------------
Top Positive Sentences:
  ('great hear trip year time age post good bill form limited coverage great season', 0.96)
  ('small defense sale size pen work pretty good hear', 0.96)
  ('love coverage production excellent excellent modern time hear great request leave hope lead regular season contract guess game playoff canadian pick canadian thing btw', 0.96)
  ('baseball school people game week score run carry hope hold yesterday score high great site kick enjoy game', 0.96)
  ('kind happy smoke share good great modern put good history baseball silly totally line', 0.95)

Top Neutral Sentences:
  ('report today agreement season pick choice year deal sign round pick play playoff', 0.95)
  ('commit game originally schedule net knowledge release schedule announce', 0.94)
  ('draft choice assume play round pick entry draft notice summary begin play playoff', 0.93)
  ('hear brother week today finish reading post regard read game play week ago mention join mailing list enter score plug read rest post spring training', 0.93)
  ('note fuel fact local owner purchase back original owner remain due contract pay channel team half dozen local cross surprised deal season', 0.93)

Top Negative Sentences: 
  ('agree suck big time thing night game', 0.96)
  ('baseball time die charge wife car tree carry firearm car report hear read watch lot game writing line stupid hurt team raise question print area question action article point kill people live quote stupid', 0.95)
  ('agree screw damn agreement show hockey game lose fan quickly decision', 0.95)
  ('shit final couple year ago shit fan', 0.94)
  ('watch watch stone watch suck watch', 0.92)

Sentiment Statistics:
  Average Sentiment Score: 0.09
  Positive Ratio: 0.22
  Neutral Ratio: 0.68
  Negative Ratio: 0.1
```


## Citation
@article{ma2024cast,
  title={CAST: Corpus-Aware Self-similarity Enhanced Topic modelling},
  author={Ma, Yanan and Xiao, Chenghao and Yuan, Chenhan and van der Veer, Sabine N and Hassan, Lamiece and Lin, Chenghua and Nenadic, Goran},
  journal={arXiv preprint arXiv:2410.15136},
  year={2024}
}


## References
<a name="reference1">1</a>. Xiao, C., Long, Y., & Moubayed, N. A. (2022). On isotropy, contextualization and learning dynamics of contrastive-based sentence representation learning. arXiv preprint arXiv:2212.09170.
