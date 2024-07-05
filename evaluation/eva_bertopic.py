import time
from bertopic import BERTopic
from dataset import Dataset
# from octis.dataset.dataset import Dataset # load preprocessed data
from sklearn.datasets import fetch_20newsgroups
from evaluation_metrics import TopicCoherence, TopicDiversity
import json
from sentence_transformers import SentenceTransformer
from datasets import load_dataset # load huggingface dataset

''' 
documents = []
with open('data/reddit.txt', 'r', encoding='utf-8') as file:
    for line in file:
        documents.append(line.strip().lower())
documents = [doc for doc in documents if doc is not None]
print("length of document", len(documents)) 
dataset = Dataset(documents)
corpus = dataset.get_corpus() 

dataset = Dataset()
dataset.load_custom_dataset_from_folder("octis_dataset/Elonmusk_lemmatized")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]

dataset = Dataset()
dataset.fetch_dataset("20NewsGroup")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]
'''
# load dataset

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
dataset = Dataset(newsgroups.data)
corpus = dataset.get_corpus()
corpus = [doc for doc in corpus if doc]
documents = [" ".join(words) for words in corpus] 



# encode documents so that you don't need to encode everytime
MODEL_NAME = 'all-mpnet-base-v2' # all-mpnet-base-v2 all-MiniLM-L6-v2

embedding_model = SentenceTransformer(MODEL_NAME)
embeddings = embedding_model.encode(documents, show_progress_bar=True)


params = {"embedding_model": MODEL_NAME,
          'min_topic_size': 5,
          "verbose": True
          }

for nr_topics in [11]:
    results = []
    for j in range(3):
        start = time.time()
        model = BERTopic(**params, nr_topics = nr_topics)
        topics, probs = model.fit_transform(documents, embeddings)
        print (f"number of topics: {len(set(topics))}")
        end = time.time()
        computation_time = float(end - start)

        twords = [
                    [
                        vals[0] for vals in model.get_topic(i)[:10]
                    ]
                    for i in range(nr_topics-1)
                ]
        print (f"the length of twords: {len(twords)}")
        print (twords)
        coherence_metric = TopicCoherence(texts=corpus, topk=10, measure="c_npmi")
        coherence_score = coherence_metric.score(twords)

        diversity_metric = TopicDiversity(topk=10)
        diversity_score = diversity_metric.score(twords)
        
        scores = {}
        scores['npmi'] = coherence_score
        scores['diversity'] = diversity_score

        result = {
            "Model": MODEL_NAME,
            "Dataset Size": len(documents),
            "Params": params,
            "Scores": scores,
            "Computation Time": computation_time,
            "Topic words": twords
        }
        results.append(result)


    with open(f"20NewsGroup_Bertopic_{MODEL_NAME}_{nr_topics-1}.json", "w") as file:
        json.dump(results, file)






