import nltk
from nltk.corpus import reuters, brown, wordnet
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import statistics
import os
from sklearn.metrics.pairwise import cosine_similarity as cs
import matplotlib.pyplot as plt
import pytrec_eval
from copy import deepcopy
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.sparse import csr_matrix
from collections import defaultdict
from gensim.models import Word2Vec


nltk.download('brown')
nltk.download('reuters')
nltk.download('inaugural')
nltk.download('punkt')

def corpus_data(corpus):
    if corpus == "reuters_corpus":
        reuters_corpus = reuters.sents(categories=['crude', 'money-fx'])
        text_data = [" ".join(i) for i in reuters_corpus]
        return text_data, reuters_corpus
    else:
        brown_corpus = brown.sents(categories=['news', 'editorial'])
        text_data = [" ".join(i) for i in brown_corpus]
        return text_data, brown_corpus
    
current_directory = os.path.dirname(os.path.abspath(__file__))
filename = 'SimLex-999.txt'
simlex_file_path = os.path.join(current_directory, filename)
word_pairs = []
similarities = []

with open(simlex_file_path, 'r', encoding='utf-8') as g:
    next(g)
    for line in g:
        fields = line.strip().split('\t')
        word_pairs.append((fields[0].lower(), fields[1].lower()))
        similarities.append(float(fields[3]))

word1, word2 = zip(*word_pairs)
similarity = np.asarray(similarities)


with open(simlex_file_path, 'r', encoding='utf-8') as g:
    next(g)
    for line in g:
        fields = line.strip().split('\t')
        word_pairs.append((fields[0].lower(), fields[1].lower()))
        similarities.append(float(fields[3]))

def top_k_similar_words(word1, word2, list_sim, vocab):
    topk = defaultdict(list)
    for i in range(len(word1)):
        if word1[i] in vocab:
            topk[word1[i]].append([list_sim[i], word2[i]])
        if word2[i] in vocab:
            topk[word2[i]].append([list_sim[i], word1[i]])
    return dict(topk)

def count_invalid_top_k_words(topk):
    invalid_count = 0
    for word, sim_words in topk.items():
        if len(sim_words) != 10:
            invalid_count += 1
    return invalid_count


def get_added_words(similarity, word2, word1_list, word2_list, top_words, word, sim_list):
    added_words = []
    if word2 in word1_list:
        indices = [i for i, x in enumerate(word1_list) if x == word2]
        for ind in indices:
            if word2_list[ind] not in top_words[:, 1] and word2_list[ind] != word and word2_list[ind] not in added_words:
                added_words.append([sim_list[ind]*(similarity/max(sim_list)), word2_list[ind]])
    if word2 in word2_list:
        indices = [i for i, x in enumerate(word2_list) if x == word2]
        for ind in indices:
            if word1_list[ind] not in top_words[:, 1] and word1_list[ind] != word and word1_list[ind] not in added_words:
                added_words.append([sim_list[ind]*(similarity/max(sim_list)), word1_list[ind]])
    return added_words

def add_to_topk(topk, addedWords, addCounter, word):
    addedWords = sorted(addedWords, reverse=True)
    while len(topk[word]) < 10 and addCounter < len(addedWords):
        topk[word].append(addedWords[addCounter])
        addCounter += 1
    return topk

def extend_topk(topk, word1_list, word2_list, sim_list, vocab):
    for word in topk.keys():
        if len(topk[word]) >= 10:
            topk[word] = sorted(topk[word], reverse=True)[:10]
        else:
            topk[word] = sorted(topk[word], reverse=True)
            top_words = np.asarray(topk[word])
            addedWords = []
            for similarity, word2 in topk[word]:
                addedWords += get_added_words(similarity, word2, word1_list, word2_list, top_words, word, sim_list)
            addCounter = 0
            topk = add_to_topk(topk, addedWords, addCounter, word)
    return topk

def transitivity_rule(topk, word1, word2, similarity, vocab, render=True):
    invalid_counts = []
    invalid_counts.append(count_invalid_top_k_words(topk))
    while True:
        updated_topk = extend_topk(topk, word1, word2, similarity, vocab)
        invalid_count = count_invalid_top_k_words(updated_topk)
        invalid_counts.append(invalid_count)
        if invalid_count == invalid_counts[-2]:
            break
        topk = updated_topk

    return topk

def generate_dictionary(model_dict, ground_truth):
    model_dictionary = {}
    for key, values in model_dict.items():
        model_dictionary[key] = {value: 1 for value in values}

    ground_truth_dictionary = {}
    for key, values in ground_truth.items():
        ground_truth_dictionary[key] = {value[1]: 1 for value in values}

    return model_dictionary, ground_truth_dictionary

def search_top_similar_words(ground_truth, similarities, vocab):
    top_similar_words = {}

    vocab_array = np.asarray(vocab)

    for key in ground_truth.keys():
        key_index = None
        if key in vocab_array:
            key_index = np.where(vocab_array == key)[0][0]

        if key_index is not None:
            similarity_vector = similarities[key_index].toarray().flatten()  # Convert to dense array
            sorted_indices = np.flip(np.argsort(similarity_vector)[-11:])  # Use argsort on dense array
            sorted_indices = np.setdiff1d(sorted_indices, key_index)
            similar_words = vocab_array[sorted_indices]
            top_similar_words[key] = similar_words

    return top_similar_words

def generate_word2vec_data(model, ground_truth):
    word_vector_dict = {}
    for word in model.wv.index_to_key:
        vectors = model.wv.most_similar(word, topn=10)
        word_vector_dict[word] = {v[0]: 1 for v in vectors}

    ground_truth_dict = {}
    for word, similar_words in ground_truth.items():
        ground_truth_dict[word] = {sw[1]: 1 for sw in similar_words}

    return word_vector_dict, ground_truth_dict

def mean_ndcg(evaluation_results):
    ndcg_scores = [result['ndcg'] for result in evaluation_results.values()]
    return sum(ndcg_scores) / len(ndcg_scores)

def median_ndcg(evaluation_results):
    ndcg_scores = [result['ndcg'] for result in evaluation_results.values()]
    return np.median(ndcg_scores)

def top_k_ndcg(evaluation_results, k=10):
    ndcg_scores = [result['ndcg'] for result in evaluation_results.values()]
    top_k_scores = ndcg_scores[:k]
    return sum(top_k_scores) / len(top_k_scores)

def interquartile_range(evaluation_results):
    ndcg_scores = [result['ndcg'] for result in evaluation_results.values()]
    return np.percentile(ndcg_scores, 75) - np.percentile(ndcg_scores, 25)


word1, word2 = zip(*word_pairs)
similarity = np.asarray(similarities)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
C, _ = corpus_data("reuters_corpus")
tfidf_matrix = tfidf_vectorizer.fit_transform(C)
tfidf_dataframe = pd.DataFrame(tfidf_matrix.T.todense())
tfidf_similarities = csr_matrix(cs(tfidf_dataframe))
tfidf_vocab = list(tfidf_vectorizer.vocabulary_.keys())
top_similar_words = top_k_similar_words(word1, word2, similarity, tfidf_vocab)
transitive_similar_words = transitivity_rule(deepcopy(top_similar_words), word1, word2, similarity, tfidf_vocab, render=False)
model_tops = search_top_similar_words(transitive_similar_words, tfidf_similarities, tfidf_vocab)
query_dict, run_dict = generate_dictionary(model_tops, transitive_similar_words)
evaluator = pytrec_eval.RelevanceEvaluator(query_dict, {'ndcg'})
evaluation_results = evaluator.evaluate(run_dict)
mean_ndcg_score = mean_ndcg(evaluation_results)
median_ndcg_score = median_ndcg(evaluation_results)
top_k_ndcg_score = top_k_ndcg(evaluation_results, k=10)
interquartile_range_score = interquartile_range(evaluation_results)

print(f"Mean NDCG: {mean_ndcg_score}")
print(f"Median NDCG: {median_ndcg_score}")
print(f"Top-10 NDCG: {top_k_ndcg_score}")
print(f"Interquartile Range of NDCG: {interquartile_range_score}")

window_sizes = [1, 2, 5, 10]
vector_sizes = [10, 50, 100, 300]
iterations = 1000

# Load data
_, reut_ds = corpus_data('reuters_corpus')

# Iterate over window sizes and vector sizes
for window_size in window_sizes:
    for vector_size in vector_sizes:
        # Train Word2Vec model
        model = Word2Vec(sentences=reut_ds, vector_size=vector_size, window=window_size, epochs=iterations)

        # Extract vocabulary
        vocabulary = list(model.wv.index_to_key)

        # Get top similar words
        similar_words = top_k_similar_words(word1, word2, similarity, vocabulary)

        # Evaluate using Word2Vec
        method = 'Word2Vec'
        word_vector_dict, ground_truth_dict = generate_word2vec_data(model, similar_words)
        evaluator = pytrec_eval.RelevanceEvaluator(word_vector_dict, {'ndcg'})
        evaluation_results = evaluator.evaluate(ground_truth_dict)

        # Compute mean NDCG
        ndcg_scores = [result['ndcg'] for result in evaluation_results.values()]
        mean_ndcg = statistics.mean(ndcg_scores)

        # Compute additional metrics
        median_ndcg = statistics.median(ndcg_scores)
        top10_ndcg = statistics.mean(ndcg_scores[:10]) if len(ndcg_scores) >= 10 else 0
        interquartile_range_ndcg = statistics.quantiles(ndcg_scores, n=4)[-1] - statistics.quantiles(ndcg_scores, n=4)[0]

        # Print results
        print(f'Mean NDCG: window_size={window_size}, vector_size={vector_size}: {mean_ndcg}')
        print(f'Median NDCG: window_size={window_size}, vector_size={vector_size}: {median_ndcg}')
        print(f'Top-10 NDCG: window_size={window_size}, vector_size={vector_size}: {top10_ndcg}')
        print(f'Interquartile Range of NDCG: window_size={window_size}, vector_size={vector_size}: {interquartile_range_ndcg}')