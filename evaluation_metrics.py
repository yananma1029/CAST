from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel

class TopicCoherence():
    def __init__(self, texts=None, topk=10, processes=1, measure='c_npmi'):
        """
        Initialize metric

        Parameters
        ----------
        texts : list of documents (list of lists of strings)
        topk : how many most likely words to consider in
        the evaluation
        measure : (default 'c_npmi') measure to use.
        processes: number of processes
        other measures: 'u_mass', 'c_v', 'c_uci', 'c_npmi'
        """
        super().__init__()
        if texts is None:
            print ("please enter valid texts")
        else:
            self._texts = texts
        self._dictionary = Dictionary(self._texts)
        self.topk = topk
        self.processes = processes
        self.measure = measure

    def score(self, topics):
        """
        Retrieve the score of the metric

        Parameters
        ----------
        topics : a list of topics

        topics = [
        ["word1", "word2", "word3", ...],  # Topic 1
        ["word4", "word5", "word6", ...],  # Topic 2
        # ... more topics
        ]

        Returns
        -------
        score : coherence score
        """
        if topics is None:
            return -1
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(
                topics=topics,
                texts=self._texts,
                dictionary=self._dictionary,
                coherence=self.measure,
                processes=self.processes,
                topn=self.topk)
            return npmi.get_coherence()

class TopicDiversity():
    def __init__(self, topk=10):
        """
        td : percentage of unique words across all words in topics 
        """
        self.topk = topk

    def score(self, topics):

        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than ' + str(self.topk))
        else:
            unique_words = set()
            for topic in topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(topics))
            return td