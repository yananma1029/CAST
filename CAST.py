import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import umap.umap_ as umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import itertools
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import random
import os
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import time
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict


logger = logging.getLogger('CAST')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class Candidate():

    def __init__(self, documents,
                 mode: str = 'n_gram',
                 ngram_range = (2,4),
                 feature_1gram: int = 2000,
                 feature_mgram: int = 10000,
                 punctuation: str =  r'[?!,.\]'
                 ):
        self.corpus = np.array(documents)
        self.mode = mode

        self.ngram_range = ngram_range
        self.feature_1gram = feature_1gram
        self.feature_mgram = feature_mgram
        self.punctuation = punctuation

    def lower(self):
        text = self.corpus
        text = text.astype('U')
        text = np.char.lower(text)

        return text

    def n_gram(self):
        ngram_range = self.ngram_range
        feature_mgram = self.feature_mgram

        text = self.lower()
        vectorizer = CountVectorizer(analyzer='word', ngram_range=ngram_range, 
                                      max_features = feature_mgram, stop_words=set(stopwords.words('english')))
        vectorizer.fit(text)
        total = vectorizer.get_feature_names_out()

        return total

    def sentence_level(self):
        text = self.corpus
        punctuation = self.punctuation
        sentence_list = []
        for index, doc in enumerate(text):
          doc = doc.strip()
          #doc = re.split(punctuation,doc)
          doc = re.split(r'[{}]'.format(punctuation), doc)
          doc = [sent.strip() for sent in doc]
          doc = list(filter(None, doc))
          sentence_list.extend(doc)
        return list(set(sentence_list))

    def build_vocab(self):
        if self.mode == 'n_gram':
             total = self.n_gram()
        elif self.mode == 'sentence_level':
             total = self.sentence_level()
        elif self.mode == 'document_level':
             total = self.document_level()

        return total
    

class CAST:

    """
    CAST: Corpus-Aware Self-similarity Enhanced Topic Modeling, a novel topic modeling method 
    that leverages word embeddings contextualized on the dataset and self-similarity scores to 
    identify contextually meaningful topic words.

    Parameters
    ----------
    documents: List[str]
        List of documents to analyze.

    model_name: str
        Name of the pre-trained model to use for encoding.

    min_count: int, optional
        Minimum count for a word to be considered. Default is 50.

    self_sim_threshold: float (Default 0.3)
        Self-similarity threshold for word filtering. 

        Empirically, 194 we found that self-similarity scores between 0.2 and 0.4 achieved 
        better coherence and diversity scores. Excessively high thresholds may filter out 
        meaningful words, reducing topic coherence. 

    chunk_size: int (Default 1000)
        Size of document chunks for processing.

    batch_size: int (Default 32)
        Batch size for encoding. 

    candidate_mode: str (Default 'word_level')
        Mode for candidate generation ('word_level' or 'ngrams'). 

    nr_topics: int (optional)
        Number of topics to extract. Default is None (extract all).

    n_dimensions: int (Default 5)
        Number of dimensions for dimensionality reduction.

    min_cluster_size: int (Default 5)
        Minimum cluster size for HDBSCAN.  
        
        You can adjust the number according to the dataset size, with larger size bigger value. 

    umap_args: dict, optional
        Arguments for UMAP dimensionality reduction. Default includes 'n_neighbors': 15 and 'metric': 'cosine'.

    """


    def __init__(self, documents, model_name, min_count = 50, self_sim_threshold=0.3, chunk_size=1000, batch_size=32, candidate_mode = 'word_level',
                nr_topics = None, n_dimensions = 5, min_cluster_size=5, umap_args={'n_neighbors': 15,  'metric': 'cosine'}):
        
        self.corpus = documents
        self.self_sim_threshold = self_sim_threshold
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dimension = n_dimensions
        self.min_cluster_size = min_cluster_size
        self.umap_args = umap_args
        self.umap_args['n_components'] = n_dimensions
        self.stop_words = set(stopwords.words('english'))
        self.min_count = min_count
        self.nr_topics = nr_topics
        self.candidate_mode = candidate_mode
        self.chunk_size = chunk_size


    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on the model output.

        Parameters
        ----------
        model_output: torch.Tensor
            Output from the transformer model.
        attention_mask: torch.Tensor
            Attention mask for the input.

        Returns
        -------
        torch.Tensor
            Mean pooled embeddings.
        """

        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encoding(self, documents):
        
        """
        Encode the input documents using the pre-trained model.

        Parameters
        ----------
        documents: List[str]
            List of documents to encode.

        Returns
        -------
        Tuple[List[List[int]], np.ndarray, np.ndarray]
            Tokenized documents, token embeddings, and sentence embeddings.
        """

        max_length = 512

        tokenized_batches = []
        token_embeddings_batches = []
        sentence_embeddings_batches = []

        self.model = self.model.to(self.device) 

        for start_idx in tqdm(range(0, len(documents), self.batch_size), desc="Processing batches"):
            end_idx = start_idx + self.batch_size
            batch_documents = documents[start_idx:end_idx]


            encoded_input = self.tokenizer(batch_documents, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}  
            tokenized_batch = encoded_input['input_ids'].tolist()

            with torch.no_grad():
                model_output = self.model(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
                token_embeddings_batch = model_output.last_hidden_state

            sentence_embeddings_batch = self.mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings_batch = F.normalize(sentence_embeddings_batch, p=2, dim=1)
            
            if self.device.type == 'cuda':
                sentence_embeddings_batch = sentence_embeddings_batch.cpu().numpy()
                token_embeddings_batch = token_embeddings_batch.cpu().numpy()
            else:
                sentence_embeddings_batch = sentence_embeddings_batch.numpy()
                token_embeddings_batch = token_embeddings_batch.numpy() 

            tokenized_batches.extend(tokenized_batch)
            token_embeddings_batches.append(token_embeddings_batch)  
            sentence_embeddings_batches.append(sentence_embeddings_batch)

        tokenized = tokenized_batches
        token_embeddings = np.concatenate(token_embeddings_batches, axis=0)
        sentence_embeddings = np.concatenate(sentence_embeddings_batches, axis=0)

        return (tokenized, token_embeddings, sentence_embeddings)


    def build_word_embeddings(self, tokenized, token_embeddings):
        """
        Build word embeddings from tokenized documents and token embeddings.

        Unlike static encoding methods, our approach dynamically reconstructed word embeddings from token embeddings generated 
        by the model's last hidden state. This process involved identifying subword tokens (e.g., those starting with \#\#) 
        and concatenating them with preceding tokens to form complete words. The word embedding was then calculated by averaging the subword embeddings.

        Parameters
        ----------
        tokenized: List[List[int]]
            Tokenized documents.
        token_embeddings: np.ndarray
            Token-level embeddings.

        Returns
        -------
        List[Dict[str, List[np.ndarray]]]
            List of dictionaries mapping words to their embeddings for each document.
            
            For example:
            Sen 1: [{I: [vec1,vec2]}, {Love: [vec1]}, {playing: [vec1,vec2,vec3]}, {football: [vec1]...}], 
            Sen 2: [{I: [vec1]}, {like: [vec1,vec2]}, {NLP: [vec1]...]
            ...
        """

        word_embedding_list = []

        for sen_index, sen in tqdm(enumerate(tokenized), total=len(tokenized), desc="Building word embeddings"):
            current_word = ""
            current_word_embeddings = []
            word_embeddings = {}

            for token_index, token_id in enumerate(sen):
                token = self.tokenizer.decode(token_id)

                # Check if the token is a subword
                if token.startswith('##'):
                    # If it's a subword, append it to the current word and add its embedding
                    current_word += token[2:]
                    current_word_embeddings.append(token_embeddings[sen_index][token_index])
                else:
                    # If it's a new word, store the previous word's embeddings and reset
                    if current_word:
                        # Check if the word is not a stop word before adding it
                        if current_word.lower() not in self.stop_words:
                            if current_word in word_embeddings.keys():
                                word_embeddings[current_word].append(self.l2_normalize(np.mean(current_word_embeddings, axis=0)))
                            else:
                                word_embeddings[current_word] = [self.l2_normalize(np.mean(current_word_embeddings, axis=0))]

                        current_word = token
                        current_word_embeddings = [token_embeddings[sen_index][token_index]]
                    else:
                        current_word = token
                        current_word_embeddings = [token_embeddings[sen_index][token_index]]

            # Store the embeddings for the last word
            if current_word and current_word.lower() not in self.stop_words:
                if current_word in word_embeddings.keys():
                    word_embeddings[current_word].append(self.l2_normalize(np.mean(current_word_embeddings, axis=0)))
                else:
                    word_embeddings[current_word] = [self.l2_normalize(np.mean(current_word_embeddings, axis=0))]

            # Append the word_embeddings dictionary for the current sentence to word_embedding_list
            word_embedding_list.append(word_embeddings)

        return word_embedding_list 

  
    def merge_word_embeddings(self, word_embedding_list):
        """
        Merge word embeddings from different sentences. This is useful to calculate self_similarity scores across sentences.

        Parameters
        ----------
        word_embedding_list: List[Dict[str, List[np.ndarray]]]
            List of word embeddings dictionaries across sentences.

        Returns
        -------
        Dict[str, np.ndarray]
            Merged word embeddings dictionary.
        """

        merged_word_embeddings = defaultdict(list)
        
        for word_embeddings_dict in tqdm(word_embedding_list, desc="Merging word embeddings"):
            for word, embeddings_list in word_embeddings_dict.items():
                lower_word = word.lower()
                if lower_word not in self.stop_words and len(lower_word)>1:
                    merged_word_embeddings[word].extend(embeddings_list)

        for word, embeddings_list in merged_word_embeddings.items():
            if len(embeddings_list) > 10000:
                merged_word_embeddings[word] = np.array(embeddings_list[:10000])
            else:
                merged_word_embeddings[word] = np.array(embeddings_list)

        return merged_word_embeddings

    def ss_similarity(self, word_embeddings, max_sample_size=2000):

        """
        Calculate self-similarity scores for words.

        Parameters
        ----------
        word_embeddings: Dict[str, np.ndarray]
            Dictionary of word embeddings.
        max_sample_size: int, optional
            Maximum sample size for similarity calculation. Default is 2000.

        Returns
        -------
        Dict[str, float]
            Dictionary of self-similarity scores for words.
        """

        ss_score = {}
        for k, v in word_embeddings.items():
            if len(v) > max_sample_size:
                v = random.sample(list(v), max_sample_size)
            if len(v) > 1:
                v = np.array(v)
                sim_matrix = cosine_similarity(v, v)
                self_similarity = round(
                    (np.sum(sim_matrix) - len(sim_matrix)) / (len(sim_matrix) * (len(sim_matrix) - 1)), 3)
                ss_score[k] = self_similarity

        return ss_score
    
    def build_embeddings(self, candidate_mode='word_level'):
        """
        Generate sentence and word embeddings for the corpus, storing them in the 'embeddings' folder. 
        The function will first check if embeddings already exist in the 'embeddings' folder. 
        If they do, it will load the existing embeddings to save time when the model is run multiple times.

        Parameters
        ----------
        candidate_mode: str, optional
            Mode for candidate generation ('word_level' or 'ngrams'). Default is 'word_level'.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, np.ndarray]]
            Sentence embeddings and word embeddings.
        """

        embeddings_path = os.path.join(os.getcwd(), "embeddings")
        os.makedirs(embeddings_path, exist_ok=True)
        sen_embeddings_file = os.path.join(embeddings_path, f"{self.model_name.replace('/', '_')}_sen_embeddings.pkl")
        word_embeddings_file = os.path.join(embeddings_path, f"{self.model_name.replace('/', '_')}_word_embeddings.pkl")

        if os.path.exists(sen_embeddings_file) and os.path.exists(word_embeddings_file):
            logger.info(f"Loading pre-computed sentence embeddings and word embeddings")
            with open(sen_embeddings_file, "rb") as f:
                sentence_embeddings = pickle.load(f)
            with open(word_embeddings_file, "rb") as f:
                word_embeddings = pickle.load(f)
        else:
            logger.info(f"Tokenizing text with model: {self.model_name}")

            sentence_embeddings_batches = []
            word_embedding_lists = []

            for start_idx in tqdm(range(0, len(self.corpus), self.chunk_size), desc="Processing chunk batches"):
                end_idx = start_idx + self.chunk_size

                batch_documents = self.corpus[start_idx:end_idx]
                tokenized_batch, token_embeddings_batch, sentence_embeddings_batch = self.encoding(batch_documents)
                sentence_embeddings_batches.extend(sentence_embeddings_batch)  

                word_embedding_list = self.build_word_embeddings(tokenized_batch, token_embeddings_batch)              
                word_embedding_lists.extend(word_embedding_list)
                    

            sentence_embeddings = np.array(sentence_embeddings_batches)
            print (f"length of sentence_embeddings: {len(sentence_embeddings)}")
            with open(sen_embeddings_file, "wb") as f:
                pickle.dump(sentence_embeddings, f)

            word_embeddings = self.merge_word_embeddings(word_embedding_lists)
            with open(word_embeddings_file, "wb") as f:
                pickle.dump(word_embeddings, f)

        logger.info(f"Filtering out functional words and building candidate words")
        ss_score = self.ss_similarity(word_embeddings)
        candidate_words = self.build_candidates(ss_score, word_embeddings, mode=candidate_mode)

        self.sentence_embeddings = sentence_embeddings
        self.word_embeddings = candidate_words

        return self.sentence_embeddings, self.word_embeddings

    def build_ngram_embeddings(self, ngrams, word_embeddings_dict):

        """
        Build embeddings for n-grams based on word embeddings.

        Parameters
        ----------
        ngrams: List[str]
            List of n-grams.
        word_embeddings_dict: Dict[str, np.ndarray]
            Dictionary of word embeddings.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of n-gram embeddings.
        """

        word_embeddings = {}
        for k, v in word_embeddings_dict.items():
            word_embeddings[k] = self.l2_normalize(np.mean(v, axis=0))

        ngram_embeddings = {}
        for ngram in ngrams:
            words = ngram.split(' ')
            word_vecs = [word_embeddings.get(word) for word in words if word in word_embeddings]
            if word_vecs:
                ngram_embeddings[ngram] = self.l2_normalize(np.mean(word_vecs, axis=0))
            else:
                # Handle the case where no word in the n-gram is found in word_embeddings
              
                pass

        return ngram_embeddings

    def build_candidates(self, ss_score, word_embeddings, mode: str = 'word_level'):

        """
        Build candidate words or n-grams based on self-similarity scores.

        Parameters
        ----------
        ss_score: Dict[str, float]
            Dictionary of self-similarity scores.
        word_embeddings: Dict[str, np.ndarray]
            Dictionary of word embeddings.
        mode: str, optional
            Mode for candidate building ('word_level' or 'ngrams'). Default is 'word_level'.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of candidate words or n-grams and their embeddings.
        """
                
        candidates = {}
        if mode == 'word_level':
            for word, score in ss_score.items():
                if (score > self.self_sim_threshold and not word.isdigit() and len(word_embeddings[word]) >= self.min_count
                    and word.lower() not in ['<s>', '</s>', '<pad>', '[cls]', '[sep]', '[pad]'] and len(word)>1):
                    candidates[word] = self.l2_normalize(np.mean(word_embeddings[word], axis=0))

        elif mode == 'ngrams':
            logger.info(f"Building ngrams candidates")
            ngrams = Candidate(self.corpus, mode = 'n_gram').build_vocab()
            candidates = self.build_ngram_embeddings(ngrams, word_embeddings)

        return candidates 


    def DR(self, embedding, umap_args):

        """
        Perform dimensionality reduction using UMAP.

        Parameters
        ----------
        embedding: np.ndarray or torch.Tensor
            Input embeddings.
        umap_args: Dict
            Arguments for UMAP.

        Returns
        -------
        np.ndarray
            Reduced embeddings.
        """

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        reducer = umap.UMAP(**umap_args)
        reduced_embedding = reducer.fit_transform(embedding)
        return reduced_embedding

    def clustering(self, reduced_embedding, min_cluster_size):
        """
        Perform clustering using HDBSCAN.

        Parameters
        ----------
        reduced_embedding: np.ndarray
            Reduced embeddings.
        min_cluster_size: int
            Minimum cluster size for HDBSCAN.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Cluster labels and outlier scores.
        """

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(reduced_embedding)
        outlier_scores = clusterer.outlier_scores_
        return cluster_labels, outlier_scores

    def centroid(self, sen_embeddings, cluster_labels):
        """
        Calculate centroids for each cluster.

        Parameters
        ----------
        sen_embeddings: np.ndarray or torch.Tensor
            Sentence embeddings.
        cluster_labels: np.ndarray
            Cluster labels.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[int, np.ndarray]]
            Updated DataFrame with cluster information and dictionary of centroids sorted according to topic size.
        """

        documents = self.corpus
        centroids = {}
        if isinstance(sen_embeddings, torch.Tensor):
            rep = np.array(sen_embeddings)
        else:
            rep = sen_embeddings
        df = pd.DataFrame()
        df['text'] = documents
        df['label'] = cluster_labels
        df['rep'] = rep.tolist()
        #df['centroid'] = np.nan  # Initialize 'centroid' column with NaN

        # Calculate centroids for clusters
        for m in set(cluster_labels):
            if m == -1:
                continue
            index = np.where(cluster_labels == m)[0]
            centroid = self.l2_normalize(np.mean(rep[index], axis=0))
            centroids[m] = centroid

        # remove noise where label = -1
        df = df[df['label'] != -1]

        # sort centroids according to topic size
        updated_df, updated_centroids = self.reset_index(df, centroids)

        return updated_df, updated_centroids
    
    def reset_index(self, df, centroids):
        """
        Reset cluster indices based on topic size.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with cluster information.
        centroids: Dict[int, np.ndarray]
            Dictionary of centroids.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[int, np.ndarray]]
            Updated DataFrame and centroids with reset indices.
        """

        label_list = list(centroids.keys())
        topic_size = {label: len(df[df['label'] == label]) for label in label_list}
        sorted_topic_size = dict(sorted(topic_size.items(), key=lambda item: item[1], reverse=True)) 
        sorted_centroids = {key: centroids[key] for key in sorted_topic_size}        
    
        old_to_new_labels = {old_key:i for i,old_key in enumerate(sorted_centroids.keys())}
        updated_centroids = {new_label: sorted_centroids[old_label] for old_label, new_label in old_to_new_labels.items()}

        df.loc[:, 'label'] = df['label'].map(old_to_new_labels)
        df.loc[:, 'centroid'] = df['label'].map(updated_centroids).values

        return df, updated_centroids

    def l2_normalize(self, vectors):
        """
        Perform L2 normalization on vectors.

        Parameters
        ----------
        vectors: np.ndarray
            Input vectors.

        Returns
        -------
        np.ndarray
            L2 normalized vectors.
        """
        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]
    
    def get_topn_clusters (self, centroids):
        """
        Get top N clusters based on topic size.

        Parameters
        ----------
        centroids: Dict[int, np.ndarray]
            Dictionary of the sorted centroids.

        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary of top N centroids.
        """
        if self.nr_topics is None:
            return centroids

        original_num_topics = len(centroids.keys())

        if original_num_topics < 2 or self.nr_topics >= original_num_topics:
            logger.warning(f"Please enter a value that is less than the original number of topics {original_num_topics}")
            return centroids

        top_centroids = dict(list(centroids.items())[:self.nr_topics])
        return top_centroids


    def get_top_n_sentences(self, nr_sentences=5, cluster_number = None):
        """
        Retrieves the top n sentences for each cluster based on cosine similarity to the cluster centroid.
        
        Parameters:
        nr_sentences (int, default=5): Number of top sentences to retrieve for each cluster.
        cluster_number (int, optional): Specific cluster number to retrieve sentences from. Defaults to None. 
            If it is None, it will return the top sentences for all clusters.
        
        Returns:
        dict: A dictionary with cluster labels as keys and lists of top sentences as values.
        """
        
        cluster_data = []

        if cluster_number is None: 
            
            for cluster, centroid in self.top_centroids.items():
                cluster_df = self.document_vector[self.document_vector['label'] == cluster]
                sentences = cluster_df['text'].tolist()
                candidate_embeddings = cluster_df['rep'].tolist()

                # Compute cosine similarity between candidate embeddings and the centroid
                similarities = cosine_similarity([centroid], candidate_embeddings)[0]

                # Pair sentences with their similarity scores and sort by similarity in descending order
                sentence_similarity_pairs = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)

                # Select the top n sentences
                top_sentences = [sentence for sentence, _ in sentence_similarity_pairs[:nr_sentences]]

                cluster_data.append({
                    'Topic': cluster,
                    'Count': len(cluster_df),
                    'Top_Sentences': top_sentences
                })
        
        else:
            if cluster_number not in self.top_centroids:
                logger.warning("Please enter a valid cluster")
                return
            
            centroid = self.top_centroids[cluster_number]
            cluster_df = self.document_vector[self.document_vector['label'] == cluster_number]
            sentences = cluster_df['text'].tolist()
            candidate_embeddings = cluster_df['rep'].tolist()

            similarities = cosine_similarity([centroid], candidate_embeddings)[0]

            sentence_similarity_pairs = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)

            top_sentences = [sentence for sentence, _ in sentence_similarity_pairs[:nr_sentences]]

            cluster_data.append({
                'Topic': cluster_number,
                'Count': len(cluster_df),
                'Top_Sentences': top_sentences
            })

        df = pd.DataFrame(cluster_data, columns=['Topic', 'Count', 'Top_Sentences'])

        return df


    def get_topic_words(self, embeddings, centroids):
        """
        Get top words for each topic based on cosine similarity to centroids.

        Parameters
        ----------
        embeddings: Dict[str, np.ndarray]
            Dictionary of word embeddings.
        centroids: Dict[int, np.ndarray]
            Dictionary of centroids.

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping topic IDs to lists of top words for each topic.
        """
        topic_keywords = {}
        candidate_embeddings = np.array(list(embeddings.values()))
        centroids = centroids.copy()

        for cluster_id, centroid in centroids.items():
         
            similarities = cosine_similarity([centroid], candidate_embeddings)[0]
            word_sim_pairs = sorted(zip(embeddings.keys(), similarities), key=lambda x: x[1], reverse=True)
            topic_keywords[cluster_id] = [word for word, _ in word_sim_pairs[:10]]

        return topic_keywords



    def pipeline(self):
        """
        Pipeline of CAST

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping topic IDs to lists of top words for each topic.
        """
        self.sen_embeddings, self.word_embeddings = self.build_embeddings(candidate_mode= self.candidate_mode)
        if isinstance(self.sentence_embeddings, torch.Tensor):
            self.sentence_embeddings = self.sentence_embeddings.cpu()
            
        logger.info(f"Reducing embedding dimensions to {self.dimension}D")

        umap_embeddings = self.DR(np.array(self.sentence_embeddings), umap_args=self.umap_args) # used to find clusters
        candidate_words = self.word_embeddings

        logger.info("Finding dense areas of documents")
        cluster_labels, outlier_scores = self.clustering(umap_embeddings, min_cluster_size=self.min_cluster_size)

        self.document_vector, self.centroids = self.centroid(self.sentence_embeddings, cluster_labels)

        if self.nr_topics is None:
            self.top_centroids = self.centroids
        else:
            self.top_centroids = self.get_topn_clusters(self.centroids)


        logger.info("Finding topics")
        top_words = self.get_topic_words(candidate_words, self.top_centroids)

        return top_words


