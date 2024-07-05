import time
from octis.models.LDA import LDA
#from octis.dataset.dataset import Dataset
from dataset import Dataset
from evaluation_metrics import TopicCoherence, TopicDiversity
import json
from sklearn.datasets import fetch_20newsgroups

#from dataset import Dataset

''' 
dataset = Dataset()
dataset.load_custom_dataset_from_folder("octis_dataset/reddit")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]

# for LDA we imported the preprocessed 20NewsGroups from octis
dataset = Dataset()
dataset.fetch_dataset("20NewsGroup")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]
'''

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
dataset = Dataset(newsgroups.data)
corpus = dataset.get_corpus()
corpus = [doc for doc in corpus if doc]
documents = [" ".join(words) for words in corpus] 

for nr_topics in [10]:
    results = []
    for i, random_state in enumerate([0, 21, 42]):
        params = {"random_state": random_state,
                "alpha" : "auto", "eta": "auto", "decay" : 0.5, "offset" : 1.0,
                "iterations" : 400, "gamma_threshold" : 0.001}
        
        model = LDA(**params, num_topics=nr_topics)
        model.use_partitions = False


        start = time.time()
        output_tm = model.train_model(dataset)
        end = time.time()
        computation_time = end - start

        twords = output_tm['topics']
        
        coherence_metric = TopicCoherence(texts=corpus, topk=10, measure="c_npmi")
        coherence_score = coherence_metric.score(twords)

        diversity_metric = TopicDiversity(topk=10)
        diversity_score = diversity_metric.score(twords)
        
        scores = {}
        scores['npmi'] = coherence_score
        scores['diversity'] = diversity_score

        result = {
            "Dataset Size": len(documents),
            "Params": params,
            "Scores": scores,
            "Computation Time": computation_time,
            "Topic words": twords
        }
        results.append(result)


    with open(f"20NewsGroups_LDA_{nr_topics}.json", "w") as file:
        json.dump(results, file)






