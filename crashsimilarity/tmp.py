import json

import logging

from gensim.models import doc2vec

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
