import time
#from dataset import Dataset
from octis.dataset.dataset import Dataset
from evaluation_metrics import TopicCoherence, TopicDiversity
import json


dataset = Dataset()
dataset.load_custom_dataset_from_folder("octis_dataset/Elonmusk_lemmatized")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]


def read_twords(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            # Remove any leading/trailing whitespace characters
            line = line.strip()
            if line:  # If the line is not empty
                # Split the line into key and values part
                key, values = line.split(':')
                # Split the values by comma and strip any extra whitespace
                value_list = [value.strip() for value in values.split(',')]
                # Add the key and list of values to the dictionary
                result_dict[key] = value_list
    return result_dict

results = []
for nr_topics in [10, 20, 30]:
    twords_dic = read_twords(f'/mnt/iusers01/fatpou01/compsci01/p66988ym/scratch/TopicModeling/TopClus/results_Elonmusk/topics_final_{nr_topics}.txt')
    twords = list(twords_dic.values())
    coherence_metric = TopicCoherence(texts=corpus, topk=10, measure="c_npmi")
    umass = TopicCoherence(texts=corpus, topk=10, measure='u_mass')
    uci = TopicCoherence(texts=corpus, topk=10, measure='c_uci')
    
    coherence_score = coherence_metric.score(twords)
    umass_score = umass.score(twords)
    uci_score = uci.score(twords)
  
     
    diversity_metric = TopicDiversity(topk=10)
    diversity_score = diversity_metric.score(twords)

    scores = {}
    scores['npmi'] = coherence_score
    scores['diversity'] = diversity_score
    scores['umass'] = umass_score
    scores['uci'] = uci_score

    result = {
        "Dataset Size": len(corpus),
        "nr_topics" : nr_topics,
        "Scores": scores,
        "Topic words": twords
    }
    results.append(result)


with open("Elonmusk_Pre_TopClus.json", "w") as file:
    json.dump(results, file)




