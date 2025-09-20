from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from scipy.optimize import linear_sum_assignment as hungarian
from datasets import load_dataset as load_dataset_hf
import pandas as pd
import numpy as np
import json

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def avg_closest_distance(emba,embb):
    # Compute pairwise Euclidean distance matrix
    distance_matrix = cdist(emba, embb, metric='euclidean')

    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Get the assigned distances
    assigned_distances = distance_matrix[row_ind, col_ind]

    # Compute the average closest distance
    average_closest_distance = np.mean(assigned_distances)

    # Display results
    return average_closest_distance

tweet_data_path = 'data_loaders/tweet.csv'
massive_data_path = 'data_loaders/massive.jsonl'

def load_dataset(type, opt = None):
    if type == 'bank77':
        return load_bank77()
    elif type == 'clinic':
        return load_clinc()
    elif type == 'tweet':
        return load_tweet()
    elif 'massive' in type:
        return load_massive(opt)
    elif type == 'goemo':
        return load_goemo()


def load_goemo():
    instructor_prompt = 'Represent the emotion in this sentence:'
    prompt = 'The following is cluster of comments expressing the same emotion. Concisely classify the emotion(s) that summarize the following cluster:'
    text_type = "Emotion(s):"

    num_clusters = 27
    df = pd.read_csv('goemo.tsv', sep = '\t', header = None)
    df.columns = ['text','emo','user']
    num_emo = []
    for i in df.emo:
        i = i.strip()
        if ',' in i:
            num_emo.append(0)
        else:
            num_emo.append(1)
    df['num_emo'] = num_emo

    #keep only those with unique labels
    df = df[df.num_emo==1]

    #remove neutral expressions
    df = df[df.emo!='27']

    labels = [int(i) for i in df.emo]
    texts = list(df.text)
    return labels, texts, num_clusters, prompt, text_type, instructor_prompt

def load_tweet():
    num_clusters = 89
    data = pd.read_csv(tweet_data_path, sep="\t")
    return list(data['label']), list(data['text']), num_clusters

def load_clinc():
    instructor_prompt = 'Represent this question for clustering:'
    prompt = 'The following is cluster of questions. Write a question that summarizes the following cluster:'
    text_type = "Question:"
    num_clusters = 150
    dataset = load_dataset_hf("clinc_oos", "small")
    test_split = dataset["test"]
    texts = test_split["text"]
    intents = test_split["intent"]
    filtered_pairs = [(t, i) for (t, i) in zip(texts, intents) if i != 42]
    filtered_texts, filtered_intents = zip(*filtered_pairs)
    intent_mapping = {}
    for intent in filtered_intents:
        if intent not in intent_mapping:
            intent_mapping[intent] = len(intent_mapping)
    remapped_intents = [intent_mapping[i] for i in filtered_intents]

    return remapped_intents, list(filtered_texts), num_clusters, prompt, text_type, instructor_prompt


def load_bank77():
    instructor_prompt = 'Represent this online banking question for clustering:'
    prompt = 'The following is a cluster of online banking questions. Write a single question that represents the following cluster concisely:'
    text_type = 'Sentence:'

    #prompt = ""
    #text_type = ""

    num_clusters = 77
    dataset = load_dataset_hf("banking77")
    test_split = dataset["test"]
    texts = test_split["text"]
    labels = test_split["label"]

    return labels, texts, num_clusters, prompt, text_type, instructor_prompt

def load_massive(opt = 'I'):
    instructor_prompt = 'Represent this virtual assistant utterance for clustering:'
    data = []
    with open(massive_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))  # Load each JSON object
    
    df = pd.DataFrame(data)
    df = df[df.partition=='test']
    texts = list(df.utt)
    if opt == 'I':
        prompt = 'The following is a cluster of virtual assistant utterances. Write a short utterance that represents the following cluster:'
        text_type = 'Utterance:'
        num_clusters = 59
        labels = list(df.intent)
    elif opt == 'D':
        prompt = 'The following is a cluster of virtual assistant utterances. Write a summary that represents the following cluster:'
        text_type = 'Summary:'
        num_clusters = 18
        labels = list(df.scenario)

    encoder = LabelEncoder()
    numeric_labels = list(encoder.fit_transform(labels))

    return numeric_labels, texts, num_clusters, prompt, text_type, instructor_prompt

def cluster_metrics(y_true, y_pred, centroid_true, centroid_pred, summary_true, summary_pred = None):
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
  
    # ind = sklearn.utils.linear_assignment_.linear_assignment(w.max() - w)
    # row_ind, col_ind = linear_assignment(w.max() - w)
    row_ind, col_ind = hungarian(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

    #centroid-centroid distance
    cen_cen_dist = avg_closest_distance(centroid_pred, centroid_true)
    #centroid-summary distance
    cen_sum_dist = avg_closest_distance(centroid_pred, summary_true)

    if summary_pred is None:
        summary_pred = centroid_pred

    #summary-centroid distance
    sum_cen_dist = avg_closest_distance(summary_pred, centroid_true)
    #summary-summary distance
    sum_sum_dist = avg_closest_distance(summary_pred, summary_true)

    return [acc, nmi, cen_cen_dist, cen_sum_dist, sum_cen_dist, sum_sum_dist]