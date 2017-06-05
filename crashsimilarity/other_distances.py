from collections import defaultdict

import numpy as np
import time

from gensim.models import doc2vec
from sklearn.cluster import DBSCAN


def compressed_tagged_corpus(corpus, vocab):
    compressed = [[str(vocab.get(i, i)) for i in c] for c in corpus]
    compressed = [doc2vec.TaggedDocument(trace, [i]) for i, trace in enumerate(compressed)]
    return compressed


def dist_for_cluster(cluster, dist):
    rv = np.zeros((len(cluster), len(cluster)), dtype=np.double)
    for i, c in enumerate(cluster):
        for j, v in enumerate(cluster):
            rv[i, j] = dist[c, v]
    return rv


def dbscan(dist, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(dist)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters, core_samples_mask


def labels_to_clusters(labels):
    clusters = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            clusters[c].append(i)
    return clusters


def predicted_for_group(group, predicted):
    rv = defaultdict(int)
    for g in group:
        for c, points in predicted.items():
            if g in points:
                rv[c] += 1
    return rv


def calc_accuracy(predicted, true):
    total = sum([len(i) for i in predicted.values()])     
    good = sum([max(predicted_for_group(g, predicted).values() or [0]) for g in true])
    return good / float(total)


def struct_word_dist(w1, w2):
    parts1 = w1.split('::')
    parts2 = w2.split('::')
    if len(parts1) < len(parts2):
        return struct_word_dist(w2, w1)
    prefix = 0
    while prefix < len(parts2) and parts1[prefix] == parts2[prefix]:
        prefix += 1
    return 1 - float(prefix) / max(len(parts1), len(parts2))


def distance_matrix(corpus, calculator, prog=10):
    dist = np.zeros((len(corpus), len(corpus)), dtype=np.double)
    idx = []
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            idx.append((i, j))
    say = len(idx) // prog
    t = time.time()
    for s, (i, j) in enumerate(idx):
        if s and s % say == 0:
            print('{}%, {} s.'.format(s / (len(idx) * 0.01), time.time() - t))
        doc1 = corpus[i].words
        doc2 = corpus[j].words
        dist[i, j] = dist[j, i] = calculator(doc1, doc2)
    return dist


def edit_distance2(s1, s2, ins_cost=lambda a, b: 1, del_cost=lambda a, b: 1, subst_cost=lambda a, b: 1):
    if len(s1) < len(s2):
        return edit_distance2(s2, s1, ins_cost, del_cost, subst_cost)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + ins_cost(s1[i], s2[j])  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + del_cost(s2[j], s2[j - 1])  # than s2
            substitutions = previous_row[j] + (c1 != c2) * subst_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def edit_distance(s1, s2, dist):
    if len(s1) < len(s2):
        return edit_distance(s2, s1, dist)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            d = [dist(s1[i], s2[j]),  # insert
                 dist(s2[j], s2[j - 1]),  # del
                 dist(c1, c2)]  # subst
            d = [2 if i < 0 else i for i in d]
            insertions = previous_row[j + 1] + d[
                0]  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + d[1]  # than s2
            substitutions = previous_row[j] + (c1 != c2) * d[2]
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
