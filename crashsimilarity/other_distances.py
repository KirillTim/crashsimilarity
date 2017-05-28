import numpy as np


def edit_distance_matrix(corpus, calculator, prog=10):
    dist = np.zeros((len(corpus), len(corpus)), dtype=np.double)
    idx = []
    for i in range(len(corpus)):
        for j in range(i+1, luuen(corpus)):
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
            d = [dist(s1[i], s2[j]), #insert
                 dist(s2[j], s2[j-1]), #del
                 dist(c1, c2)] #subst
            d = [2 if i < 0 else i for i in d]
            insertions = previous_row[j + 1] + d[0] # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + d[1]      # than s2
            substitutions = previous_row[j] + (c1 != c2) * d[2]
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]