import json

import logging
import numpy as np
import time

from gensim.models import doc2vec

from gensim.corpora.dictionary import Dictionary
from pyemd import emd
from datetime import timedelta
from crashsimilarity.downloader import SocorroDownloader
from crashsimilarity import utils

logging.basicConfig(level=logging.INFO)


def corpus_generator(file_name):
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            j = json.loads(line)
            stack_trace = j[0]
            yield doc2vec.TaggedDocument(stack_trace, [i])


def download_save_day(i):
    save_to_dir = '../new_crashes'
    day = utils.utc_today() - timedelta(i)
    gen = SocorroDownloader().download_day_crashes(day)
    utils.write_json(utils.crashes_dump_file_path(day, 'Firefox', save_to_dir), gen)


# model = Doc2Vec.load('../data/model/dm_d200_all.model')
# corpus = list(corpus_generator('../data/new_clean_compressed.json'))


# Code modified from https://github.com/RaRe-Technologies/gensim/blob/4f0e2ae/gensim/models/keyedvectors.py#L339
def wmdistance(model, words1, words2, all_distances):
    dictionary = Dictionary(documents=[words1, words2])
    vocab_len = len(dictionary)

    # Sets for faster look-up.
    docset1 = set(words1)
    docset2 = set(words2)

    distances = np.zeros((vocab_len, vocab_len), dtype=np.double)

    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if t1 not in docset1 or t2 not in docset2:
                continue

            distances[i, j] = all_distances[model.wv.vocab[t2].index, i]

    if np.sum(distances) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        logging.info('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    # create bag of words from document
    def create_bow(doc):
        norm_bow = np.zeros(vocab_len, dtype=np.double)
        bow = dictionary.doc2bow(doc)

        for idx, count in bow:
            norm_bow[idx] = count / float(len(doc))

        return norm_bow

    bow1 = create_bow(words1)
    bow2 = create_bow(words2)

    return emd(bow1, bow2, distances)


def rwmd_distances(model, corpus, idx):
    model.init_sims(replace=True)

    words = corpus[idx].words
    words = [w for w in words if w in model]
    logging.info('words in model: {}'.format(len(words)))
    # Cos-similarity
    all_distances = np.array(1.0 -
                             np.dot(model.wv.syn0norm,
                                    model.wv.syn0norm[[model.wv.vocab[word].index for word in words]].transpose()),
                             dtype=np.double)

    # Relaxed Word Mover's Distance for selecting
    t = time.time()
    distances = []
    for doc_id in range(0, len(corpus)):
        doc_words = [model.wv.vocab[word].index for word in corpus[doc_id].words if word in model]
        if len(doc_words) != 0:
            word_dists = all_distances[doc_words]
            rwmd = max(np.sum(np.min(word_dists, axis=0)), np.sum(np.min(word_dists, axis=1)))
        else:
            rwmd = float('inf')
        distances.append((doc_id, rwmd))

    # distances.sort(key=lambda v: v[1])
    logging.info('First part done in ' + str(time.time() - t) + ' s.')
    return distances


def lcs(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = [(a[x - 1], x - 1, y - 1)] + result
            x -= 1
            y -= 1
    return result
