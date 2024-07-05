#from dataset import Dataset
from octis.dataset.dataset import Dataset # load preprocessed data
from CAST import CAST, Candidate
import time
from evaluation_metrics import TopicCoherence, TopicDiversity
import json
from datasets import load_dataset # load huggingface dataset
from sklearn.datasets import fetch_20newsgroups
import itertools


''' 
# load data from hugging face
dataset = load_dataset("SetFit/bbc-news") # sport business entertainment tech politics
train_dataset = dataset["train"]
test_dataset = dataset["test"]
bbc_news = []
for example in train_dataset:
    bbc_news.append(example["text"])
for example in test_dataset:
    bbc_news.append(example["text"])
documents = bbc_news
our_dataset = Dataset(documents)
corpus = our_dataset.get_corpus()
corpus = [doc for doc in corpus if doc]

dataset = Dataset()
dataset.load_custom_dataset_from_folder("octis_dataset/Elonmusk_lemmatized")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]

documents = []
with open('data/elonmusk.txt', 'r', encoding='utf-8') as file:
    for line in file:
        documents.append(line.strip().lower())
documents = [doc for doc in documents if doc is not None]
print("length of document", len(documents)) 
dataset = Dataset(documents)
corpus = dataset.get_corpus() 

dataset = Dataset()
dataset.fetch_dataset("20NewsGroup")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
dataset = Dataset(newsgroups.data)
corpus = dataset.get_corpus()
corpus = [doc for doc in corpus if doc]
documents = [" ".join(words) for words in corpus] 
'''


dataset = Dataset()
dataset.fetch_dataset("20NewsGroup")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # all-mpnet-base-v2 all-MiniLM-L6-v2

params = {"n_dimensions": 5, 
          "min_cluster_size": 5, 
          "min_count": 50, 
          "candidate_mode": 'word_level'
          }

for self_sim_threshold in [0.4]:
    results = []
    for j in range(1):
        start = time.time()
        model = CAST(documents=documents, model_name=MODEL_NAME, self_sim_threshold = self_sim_threshold, nr_topics=10, **params)

        twords = model.pipeline()
        end = time.time()
        computation_time = float(end - start)

        for cluster_id, keywords in twords.items():
            print(f"Cluster {cluster_id}: {', '.join(keywords)}") 

        twords_list = list(twords.values())

        scores={}
        coherence_metric = TopicCoherence(texts=corpus, topk=10, measure="c_npmi")
        coherence_score = coherence_metric.score(twords_list)

        diversity_metric = TopicDiversity(topk=10)
        diversity_score = diversity_metric.score(twords_list)


        print(f"TC: {coherence_score}; TD: {diversity_score};")
        scores['npmi'] = coherence_score
        scores['diversity'] = diversity_score


        result = {
            "Model": MODEL_NAME,
            "Dataset Size": len(documents),
            "nr_topics": len(twords_list),
            "Params": params,
            "Scores": scores,
            "Computation Time": computation_time,
            "Topic words": twords_list,
        }
        results.append(result)


    with open(f"20News_T_{self_sim_threshold}_Contextualized_{MODEL_NAME.split('/')[1]}_{10}.json", "w") as file:
        json.dump(results, file) 







