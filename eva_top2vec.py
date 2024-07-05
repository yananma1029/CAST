import time
from top2vec import Top2Vec
from dataset import Dataset
#from octis.dataset.dataset import Dataset
from evaluation_metrics import TopicCoherence, TopicDiversity
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import json
from datasets import load_dataset # load huggingface dataset
#from octis.evaluation_metrics.diversity_metrics import TopicDiversity
#from octis.evaluation_metrics.coherence_metrics import Coherence

''' 
dataset = Dataset()
dataset.load_custom_dataset_from_folder("octis_dataset/Elonmusk_lemmatized")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]

documents = bbc_news
dataset = Dataset(documents)
corpus = dataset.get_corpus() 

dataset = Dataset()
dataset.fetch_dataset("BBC_News")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]


'''

documents = []
with open('data/elonmusk.txt', 'r', encoding='utf-8') as file:
    for line in file:
        documents.append(line.strip().lower())
documents = [doc for doc in documents if doc is not None]
print("length of document", len(documents)) 
dataset = Dataset(documents)
corpus = dataset.get_corpus() 

MODEL_NAME = 'all-MiniLM-L6-v2'  #  all-MiniLM-L6-v2 

# Top2Vec support callable embedding_model (all-mpnet-base-v2)
def embed_with_mpnet(texts):
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(texts)

params = {"umap_args": {"n_components": 5},
          "hdbscan_args": {'min_cluster_size': 15},
          "min_count": 50
          }

for num_topics in [5]:
    results = []
    for i in range(3):
        print (f"num_topic: {num_topics}")
        start = time.time()
        model = Top2Vec(documents, embedding_model=MODEL_NAME, **params)
        topic_words, _, _ = model.get_topics()

        _ = model.hierarchical_topic_reduction(num_topics)
        topic_words, _, _ = model.get_topics(num_topics=num_topics, reduced =True)
        end = time.time()
        computation_time = float(end - start)
        twords = [topic_word[:10].tolist() for topic_word in topic_words]
        
        coherence_metric = TopicCoherence(texts=corpus, topk=10, measure="c_npmi")
        coherence_score = coherence_metric.score(twords)

        diversity_metric = TopicDiversity(topk=10)
        diversity_score = diversity_metric.score(twords)
        
        scores = {}
        scores['npmi'] = coherence_score
        scores['diversity'] = diversity_score

        result = {
            "Model": MODEL_NAME,
            "num_topic": len(topic_words),
            "Dataset Size": len(documents),
            "Params": params,
            "Scores": scores,
            "Computation Time": computation_time,
            "Topic words": twords
        }
        results.append(result)


    with open(f"Elonmusk_Top2vec_{MODEL_NAME}_{num_topics}.json", "w") as file:
        json.dump(results, file)






